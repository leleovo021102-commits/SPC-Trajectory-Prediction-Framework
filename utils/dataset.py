import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import cfg


class DeepAccidentDataset(Dataset):
    """
    DeepAccident 数据集加载器 (Agent-Centric 版)

    功能:
    1. 加载 data_process.py 生成的 .pkl 文件 (Agent-Centric 坐标系)。
    2. 加载 scalers.pkl 用于输入特征的归一化。
    3. __getitem__ 返回:
       - hist_norm: 归一化后的历史轨迹 (供 System 1 Encoder 使用)
       - map_feat:  地图特征 (通常已经是局部坐标，可视情况归一化)
       - fut_physical: 未归一化的未来轨迹 (真值，供 Loss 计算)
       - gt_action: 意图标签 (用于 Teacher Forcing)
       - raw_hist:  未归一化的历史轨迹 (供 Decoder 物理积分初始状态使用)
    """

    def __init__(self, mode='train', mock=False):
        """
        Args:
            mode: 'train' or 'val'
            mock: If True, generate random data for testing (bypass file loading)
        """
        self.mode = mode
        self.mock = mock

        if self.mock:
            print("[Dataset] Warning: Running in MOCK mode. Using random data.")
            self._create_mock_data()
        else:
            self._load_real_data()

    def _load_real_data(self):
        # 1. 检查并加载数据
        if not os.path.exists(cfg.PROCESSED_DATA_PATH):
            raise FileNotFoundError(
                f"Processed data not found at {cfg.PROCESSED_DATA_PATH}. Please run data_process.py first.")

        print(f"[Dataset] Loading dataset from {cfg.PROCESSED_DATA_PATH}...")
        with open(cfg.PROCESSED_DATA_PATH, 'rb') as f:
            self.samples = pickle.load(f)

        # 2. 加载归一化参数 (Mean/Std)
        if not os.path.exists(cfg.SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found at {cfg.SCALER_PATH}.")

        print(f"[Dataset] Loading scalers from {cfg.SCALER_PATH}...")
        with open(cfg.SCALER_PATH, 'rb') as f:
            self.scalers = pickle.load(f)

        # 将 Scaler 转换为 PyTorch Tensor
        self.pos_mean = torch.FloatTensor(self.scalers['pos_mean'])
        self.pos_std = torch.FloatTensor(self.scalers['pos_std'])
        self.vel_mean = torch.FloatTensor(self.scalers['vel_mean'])
        self.vel_std = torch.FloatTensor(self.scalers['vel_std'])

    def _create_mock_data(self):
        """生成符合 Agent-Centric 特征的随机数据用于代码调试"""
        self.samples = []
        # 生成 100 个假样本
        for i in range(100):
            # Agent-Centric: T=0 (index=29) 时刻位置必须为 (0,0)
            # 构造一个匀速直线运动: v = 10 m/s, dt = 0.1s
            # Hist: t = -2.9s ~ 0s
            hist = np.zeros((cfg.OBS_LEN, cfg.INPUT_DIM), dtype=np.float32)
            time_hist = np.linspace(-2.9, 0, cfg.OBS_LEN)
            hist[:, 0] = 10.0 * time_hist  # x = vt
            hist[:, 1] = 0.0  # y = 0
            hist[:, 2] = 10.0  # vx = 10
            hist[:, 3] = 0.0  # vy = 0

            # Future: t = 0.1s ~ 2.0s
            fut = np.zeros((cfg.PRED_LEN, 4), dtype=np.float32)
            time_fut = np.linspace(0.1, 2.0, cfg.PRED_LEN)
            fut[:, 0] = 10.0 * time_fut  # x = vt
            fut[:, 1] = 0.0  # y = 0
            fut[:, 2] = 10.0  # vx = 10
            fut[:, 3] = 0.0  # vy = 0

            # Map: 随机生成 (假设也是局部坐标)
            map_feat = np.random.randn(cfg.MAP_MAX_LINES, cfg.MAP_POINTS_PER_LINE, cfg.MAP_DIM).astype(np.float32)

            self.samples.append({
                'hist': hist,
                'future': fut,
                'map': map_feat,
                'meta': {'town': 'MockTown', 'is_accident': False}
            })

        # Mock Scalers (Identity transform for testing or simple scaling)
        self.pos_mean = torch.zeros(2)
        self.pos_std = torch.ones(2) * 10.0  # 假设 std 稍微大一点
        self.vel_mean = torch.zeros(2)
        self.vel_std = torch.ones(2) * 5.0

    def _normalize(self, traj):
        """
        归一化轨迹数据: (Value - Mean) / Std
        Input: [T, D]
        """
        traj_norm = traj.clone()
        # 归一化位置 (x, y)
        traj_norm[:, 0:2] = (traj[:, 0:2] - self.pos_mean) / self.pos_std
        # 归一化速度 (vx, vy)
        traj_norm[:, 2:4] = (traj[:, 2:4] - self.vel_mean) / self.vel_std
        return traj_norm

    def _get_gt_action(self, hist, fut):
        """
        [Teacher Forcing 核心] 基于未来轨迹生成 'Ground Truth' 意图标签。
        假设数据已经是 Agent-Centric 的 (当前时刻朝向为 X轴正向)。

        Action Space:
        0: Keep, 1: Acc, 2: Dec
        3: Left, 4: Left_Dec, 5: Left_Acc
        6: Right, 7: Right_Dec, 8: Right_Acc
        9: Stop
        """
        # 1. 纵向判定 (Longitudinal)
        # 获取当前速度 (vx)
        curr_vel = hist[-1, 2]
        # 获取未来平均速度
        fut_vel_avg = fut[:, 2].mean()

        ACC_THRESH = 1.0  # m/s difference
        STOP_VEL_THRESH = 0.5  # m/s

        # 特殊判定: Stop
        if curr_vel < STOP_VEL_THRESH and fut_vel_avg < STOP_VEL_THRESH:
            return 9  # Stop

        long_action = 0  # Keep
        vel_diff = fut_vel_avg - curr_vel

        if vel_diff > ACC_THRESH:
            long_action = 1  # Acc
        elif vel_diff < -ACC_THRESH:
            long_action = 2  # Dec

        # 2. 横向判定 (Lateral)
        # 计算未来轨迹终点相对于起点的角度 (在 Agent-Centric 下，起点是 (0,0))
        end_pos = fut[-1, :2]
        # atan2(y, x)
        angle = torch.atan2(end_pos[1], end_pos[0])

        TURN_THRESH_RAD = 0.15  # ~8.5 degrees (稍微放宽一点)

        lat_action_base = 0  # Straight

        if angle > TURN_THRESH_RAD:
            lat_action_base = 3  # Left Base ID
        elif angle < -TURN_THRESH_RAD:
            lat_action_base = 6  # Right Base ID

        # 3. 组合 ID
        # 简单的加法逻辑需要确保 ID 映射表 (config.py) 是结构化的
        # 0,1,2 (Straight) | 3,4,5 (Left) | 6,7,8 (Right)
        # 映射关系调整为: Base + Offset
        # Straight (0) + [Keep(0), Acc(1), Dec(2)] -> 0, 1, 2
        # Left (3)     + [Keep(0), Dec(1), Acc(2)] -> 注意 config 里的顺序

        # 让我们查一下 config.py 的定义:
        # 3: Left_Keep, 4: Left_Dec, 5: Left_Acc
        # 6: Right_Keep, 7: Right_Dec, 8: Right_Acc

        # 为了匹配 config，我们需要稍微调整一下 offset 逻辑
        final_id = 0

        if lat_action_base == 0:  # Straight
            final_id = 0 + long_action  # 0, 1, 2
        elif lat_action_base == 3:  # Left
            if long_action == 0:
                final_id = 3  # Keep
            elif long_action == 2:
                final_id = 4  # Dec (注意: Config里 4 是 Dec)
            elif long_action == 1:
                final_id = 5  # Acc
        elif lat_action_base == 6:  # Right
            if long_action == 0:
                final_id = 6  # Keep
            elif long_action == 2:
                final_id = 7  # Dec
            elif long_action == 1:
                final_id = 8  # Acc

        return final_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 转换为 Tensor
        hist = torch.FloatTensor(sample['hist'])  # [Obs_Len, 7]
        fut = torch.FloatTensor(sample['future'])  # [Pred_Len, 4]
        map_feat = torch.FloatTensor(sample['map'])  # [Lines, Points, D]

        # 1. 归一化 (Input for Neural Network)
        # 只归一化历史轨迹，地图如果已经是局部坐标且数值不大，可以不归一化或简单缩放
        hist_norm = self._normalize(hist)

        # 2. Target: Physical (真实米制单位)
        # [关键] 保持未来轨迹为物理数值，供 Loss 计算和物理误差评估
        fut_physical = fut

        # 3. 生成 Action Label
        action_id = self._get_gt_action(hist, fut)

        # 4. 构造返回字典
        return {
            'hist_norm': hist_norm,  # 归一化历史 (Model Input)
            'map_feat': map_feat,  # 地图特征 (Model Input)
            'fut_physical': fut_physical,  # 物理未来 (Loss Target)
            'gt_action': torch.tensor(action_id, dtype=torch.long),  # 动作标签
            'raw_hist': hist,  # 原始物理历史 (LLM Prompt Input & Decoder Start)
            'meta': sample['meta']  # 元数据 (Scene Info)
        }


# ==========================================
# Self-Testing Module
# ==========================================
if __name__ == "__main__":
    print("=== Testing Dataset Module ===")

    # Mock Test
    dataset = DeepAccidentDataset(mock=True)
    sample = dataset[0]

    print(f"Hist Norm Shape: {sample['hist_norm'].shape}")
    print(f"Fut Phys Shape:  {sample['fut_physical'].shape}")
    print(f"GT Action:       {sample['gt_action'].item()}")

    # 验证简单的匀速直线逻辑
    # Mock数据是匀速直线，应该对应 Action 0 (Straight_Keep)
    # 或者如果生成的时候有微小误差，可能是 1 或 2
    print(f"Action Validity Check: {sample['gt_action'].item() in [0, 1, 2]}")

    print("✅ Dataset Module Ready")