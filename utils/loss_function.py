import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class SafetyLoss(nn.Module):
    """
    安全约束多目标损失函数，严格对齐论文公式(17)-(20)
    包含4个核心项：回归损失、意图一致性损失、动力学可行性损失、场景合规性损失
    (原名 SafetyConstrainedLoss，为保持兼容性命名为 SafetyLoss)
    """

    def __init__(self):
        super().__init__()
        self.reg_loss_fn = nn.SmoothL1Loss(reduction='none')
        # 加载论文定义的权重 (兼容旧配置)
        self.weights = getattr(cfg, 'LOSS_WEIGHTS', {
            'reg': 1.0,
            'intent': 0.5,
            'phy': 0.1,
            'scene': 0.1
        })
        # 车辆动力学极限参数
        self.max_acc = 8.0  # m/s²
        self.max_jerk = 20.0  # m/s³
        self.max_angular_vel = 0.6  # rad/s

    def _regression_loss(self, pred_traj, gt_traj):
        """轨迹回归损失，对齐论文基础精度要求"""
        # 同时监督位置和速度
        pos_loss = self.reg_loss_fn(pred_traj[..., :2], gt_traj[..., :2]).mean()
        vel_loss = self.reg_loss_fn(pred_traj[..., 2:4], gt_traj[..., 2:4]).mean()
        return pos_loss + vel_loss

    def _intent_consistency_loss(self, pred_traj, action_ids):
        """意图一致性损失，对齐论文公式(18)，强制轨迹符合LLM输出的元动作"""
        loss_intent = torch.tensor(0.0, device=pred_traj.device)
        B, T, _ = pred_traj.shape

        # 1. Stop动作约束 (ID=9)：末端速度必须接近0
        is_stop = (action_ids == 9)
        if is_stop.any():
            final_vel = torch.norm(pred_traj[is_stop, -5:, 2:4], dim=-1)
            loss_stop = final_vel.mean()
            loss_intent += loss_stop

        # 2. 减速动作约束 (ID=2,4,7)：速度必须递减
        is_decel = (action_ids == 2) | (action_ids == 4) | (action_ids == 7)
        if is_decel.any():
            vels = torch.norm(pred_traj[is_decel, :, 2:4], dim=-1)
            acc = vels[:, 1:] - vels[:, :-1]
            # 惩罚加速行为 (acc > 0)
            loss_decel = F.relu(acc).mean()
            loss_intent += loss_decel

        # 3. 加速动作约束 (ID=1,5,8)：速度必须递增
        is_accel = (action_ids == 1) | (action_ids == 5) | (action_ids == 8)
        if is_accel.any():
            vels = torch.norm(pred_traj[is_accel, :, 2:4], dim=-1)
            acc = vels[:, 1:] - vels[:, :-1]
            # 惩罚减速行为 (acc < 0)
            loss_accel = F.relu(-acc).mean()
            loss_intent += loss_accel

        # 4. 转向动作约束 (ID=3,4,5,6,7,8)：横向偏移必须符合转向方向
        # Agent-Centric 下，y > 0 为左，y < 0 为右 (假设标准坐标系)

        # 左转 (ID 3,4,5)
        is_left_turn = (action_ids >= 3) & (action_ids <= 5)
        if is_left_turn.any():
            final_lat_offset = pred_traj[is_left_turn, -1, 1]
            # 左转必须有正的横向偏移，惩罚反向偏移
            loss_left = F.relu(-final_lat_offset).mean()
            loss_intent += loss_left

        # 右转 (ID 6,7,8)
        is_right_turn = (action_ids >= 6) & (action_ids <= 8)
        if is_right_turn.any():
            final_lat_offset = pred_traj[is_right_turn, -1, 1]
            # 右转必须有负的横向偏移，惩罚反向偏移
            loss_right = F.relu(final_lat_offset).mean()
            loss_intent += loss_right

        return loss_intent

    def _physics_feasibility_loss(self, pred_traj):
        """动力学可行性损失，对齐论文公式(19)，惩罚不符合车辆动力学的轨迹"""
        # pred_traj: [B, T, 4] (x, y, vx, vy)
        vel = pred_traj[..., 2:4]

        # 加速度 a = dv/dt (dt=0.1)
        acc = (vel[:, 1:] - vel[:, :-1]) / 0.1

        # 加加速度 jerk = da/dt
        jerk = (acc[:, 1:] - acc[:, :-1]) / 0.1

        # 航向角与角速度
        # yaw = atan2(vy, vx)
        yaw = torch.atan2(pred_traj[..., 3], pred_traj[..., 2] + 1e-6)
        yaw_diff = yaw[:, 1:] - yaw[:, :-1]
        # 角度归一化到 [-pi, pi]
        yaw_diff = (yaw_diff + torch.pi) % (2 * torch.pi) - torch.pi
        angular_vel = yaw_diff / 0.1

        # 惩罚超极限的加速度、加加速度、角速度
        loss_acc = F.relu(torch.norm(acc, dim=-1) - self.max_acc).mean()
        loss_jerk = F.relu(torch.norm(jerk, dim=-1) - self.max_jerk).mean()
        loss_angular = F.relu(torch.abs(angular_vel) - self.max_angular_vel).mean()

        return loss_acc + loss_jerk + loss_angular

    def _scene_compliance_loss(self, pred_traj, map_feat):
        """场景合规性损失，对齐论文公式(20)，惩罚车道边界越界"""
        loss_scene = torch.tensor(0.0, device=pred_traj.device)
        B, T, _ = pred_traj.shape

        # 1. 车道边界约束：惩罚超出车道范围的轨迹
        if map_feat is not None:
            # 提取车道中心线 (假设前两维是 x, y)
            # map_feat: [B, L, P, D]
            lane_centers = map_feat[..., :2]  # [B, L, P, 2]
            lane_width = 3.5  # 标准车道宽度

            # 简化计算：对每个轨迹点，找到最近的车道中心线距离
            # 注意：这步计算量较大，实际训练中可能需要优化或只采样部分点
            # 这里为了演示原理保留全量计算

            # [B, T, 1, 1, 2]
            traj_points = pred_traj[..., :2].unsqueeze(2).unsqueeze(2)

            # dists: [B, T, L, P]
            dists = torch.norm(traj_points - lane_centers.unsqueeze(1), dim=-1)

            # 找到最近的距离: min over P then min over L
            min_dist_to_lane = dists.min(dim=-1)[0].min(dim=-1)[0]  # [B, T]

            # 惩罚超出车道宽度的部分 (距离 > width/2)
            loss_lane = F.relu(min_dist_to_lane - lane_width / 2.0).mean()
            loss_scene += loss_lane

        return loss_scene

    def forward(self, pred_traj, gt_traj, action_ids, map_feat=None):
        """
        前向传播计算总损失
        Args:
            pred_traj: [B, T, 4] 预测轨迹 (x, y, vx, vy)
            gt_traj: [B, T, 4] 真值轨迹
            action_ids: [B] 元动作ID
            map_feat: [B, L, P, D] 地图特征 (可选)
        Returns:
            total_loss: 总损失标量
            loss_dict: 各分项损失字典
        """
        # 计算各分项损失
        loss_reg = self._regression_loss(pred_traj, gt_traj)
        loss_intent = self._intent_consistency_loss(pred_traj, action_ids)
        loss_phy = self._physics_feasibility_loss(pred_traj)

        # 如果提供了 map_feat，则计算场景损失
        if map_feat is not None:
            loss_scene = self._scene_compliance_loss(pred_traj, map_feat)
        else:
            loss_scene = torch.tensor(0.0, device=pred_traj.device)

        # 加权求和总损失
        total_loss = (
                self.weights['reg'] * loss_reg +
                self.weights['intent'] * loss_intent +
                self.weights['phy'] * loss_phy +
                self.weights['scene'] * loss_scene
        )

        # 损失字典，用于日志记录
        loss_dict = {
            'loss': total_loss.item(),  # Compatible key for train.py
            'total_loss': total_loss.item(),
            'reg': loss_reg.item(),
            'intent': loss_intent.item(),
            'phy': loss_phy.item(),
            'scene': loss_scene.item()
        }

        return total_loss, loss_dict


# ==========================================
# Self-Testing Module
# ==========================================
if __name__ == "__main__":
    print("=== Testing SafetyLoss Module ===")

    criterion = SafetyLoss()
    B, T = 4, cfg.PRED_LEN

    # Mock Data
    pred = torch.randn(B, T, 4, requires_grad=True)
    gt = torch.randn(B, T, 4)
    action_ids = torch.tensor([0, 9, 1, 2])
    map_feat = torch.randn(B, cfg.MAP_MAX_LINES, cfg.MAP_POINTS_PER_LINE, cfg.MAP_DIM)

    print("\n[Test] Forward Pass with Map...")
    loss, loss_dict = criterion(pred, gt, action_ids, map_feat)

    print(f"Total Loss: {loss.item():.4f}")
    print("Metrics Break-down:")
    for k, v in loss_dict.items():
        print(f"  - {k}: {v:.4f}")

    print("\n[Test] Backward Pass...")
    loss.backward()
    if pred.grad is not None:
        print("✅ Gradients computed successfully.")
    else:
        print("❌ Gradient flow failed.")