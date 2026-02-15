import torch
import torch.nn as nn
from config import cfg
from encoders import AgentEncoder, MapEncoder, InteractionModule
from llm_reasoning import ActionEmbedding


class GatedIntentionFusion(nn.Module):
    """
    门控融合模块：动态调节 System 1 (Context) 和 System 2 (Intention) 的权重。
    论文核心思想：Semantic-Physical Closed Loop
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.intent_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.out_proj = nn.LayerNorm(hidden_dim)

    def forward(self, context, intent):
        # context: [B, H], intent: [B, H]
        combined = torch.cat([context, intent], dim=-1)
        z = self.gate(combined)
        # 加权融合: z 控制 intent 的注入程度
        fused = self.context_proj(context) * (1 - z) + self.intent_proj(intent) * z
        return self.out_proj(fused)


class ResidualTrajectoryDecoder(nn.Module):
    """
    [重构核心] 基于残差学习的轨迹解码器 (MLP-Based / Non-Autoregressive)

    优势：
    1. Agent-Centric 友好：初始位置恒为 (0,0)，模型只需预测相对位移。
    2. 无累积误差：一次性输出所有时间步。
    3. 物理先验：零初始化使得初始行为等于 CV (匀速) 模型。
    """

    def __init__(self):
        super().__init__()
        self.hidden_dim = cfg.HIDDEN_DIM
        self.pred_len = cfg.PRED_LEN

        self.fusion = GatedIntentionFusion(cfg.HIDDEN_DIM)

        # 状态编码：将物理初始速度 [vx, vy] 映射进特征空间
        # 在 Agent-Centric 坐标系下，vy 接近 0，vx 是标量速度
        self.state_embed = nn.Sequential(
            nn.Linear(2, cfg.HIDDEN_DIM),
            nn.LayerNorm(cfg.HIDDEN_DIM),
            nn.ReLU()
        )

        # 解码主干 (MLP)
        self.decoder_net = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            # Output: [T * 2] -> 预测 T 个时刻的加速度 (ax, ay)
            nn.Linear(512, self.pred_len * 2)
        )

        # [关键技巧] 零初始化输出层
        # 训练初始阶段，Residual Acc 为 0 -> 预测轨迹 = 匀速直线运动
        with torch.no_grad():
            self.decoder_net[-1].weight.fill_(0)
            self.decoder_net[-1].bias.fill_(0)

    def forward(self, context_feat, intent_emb, current_state_phys):
        """
        Args:
            current_state_phys: [B, 7] Agent-Centric 物理状态
                                pos=(0,0), vel=(v, 0)
        """
        B = context_feat.size(0)

        # 1. Fusion
        scene_feat = self.fusion(context_feat, intent_emb)  # [B, H]

        # 2. State Embedding
        # 在 Agent-Centric 坐标系下：
        # curr_pos 恒为 [0, 0] (不需要作为特征输入)
        # curr_vel 主要是 [v, 0] (包含了速度大小信息)
        curr_vel = current_state_phys[:, 2:4]  # [B, 2]

        # Input scaling: 速度通常 < 30m/s，除以 20 归一化到 ~1
        vel_in = curr_vel / 20.0
        state_feat = self.state_embed(vel_in)  # [B, H]

        # [B, 2H]
        decode_input = torch.cat([scene_feat, state_feat], dim=-1)

        # 3. 预测残差 (Acceleration)
        # [B, T*2] -> [B, T, 2]
        acc_pred = self.decoder_net(decode_input).view(B, self.pred_len, 2)

        # Restore Scale: 网络输出 [-1, 1], 还原为加速度 [-5, 5] m/s^2
        acc_phys = acc_pred * 5.0

        # 4. 物理积分 (Vectorized Physics Integration)
        # P_t = P_0 + V_0 * t + \int \int a dt
        dt = 0.1

        # 积分速度: v_t = v_0 + cumsum(a * dt)
        vel_deltas = acc_phys * dt
        # cumsum dim=1 (Time dimension)
        pred_vels = curr_vel.unsqueeze(1) + torch.cumsum(vel_deltas, dim=1)

        # 积分位置: p_t = p_0 + cumsum(v_t * dt)
        # 注意: p_0 在 Agent-Centric 下是 (0,0)
        # 所以直接累加速度即可得到相对位移
        pos_deltas = pred_vels * dt
        pred_pos = torch.cumsum(pos_deltas, dim=1)  # [B, T, 2]

        # 拼接: [B, T, 4] (x, y, vx, vy)
        pred_traj = torch.cat([pred_pos, pred_vels], dim=-1)

        return pred_traj


class HeterogeneousPredictor(nn.Module):
    """
    完整的预测模型架构
    """

    def __init__(self):
        super().__init__()
        self.agent_enc = AgentEncoder()
        self.map_enc = MapEncoder()
        self.interaction = InteractionModule()
        self.action_embed = ActionEmbedding()
        self.decoder = ResidualTrajectoryDecoder()

    def forward(self, hist_norm, map_feat, action_ids, raw_hist):
        """
        Args:
            hist_norm: [B, T, 7] 归一化后的历史轨迹
            map_feat:  [B, L, P, 7] 地图特征 (Agent-Centric 物理坐标)
            action_ids: [B] 意图 ID
            raw_hist:  [B, T, 7] 原始物理历史 (用于提取初始速度)
        """
        # 1. Encoding Phase (Normalized Inputs)
        agent_emb = self.agent_enc(hist_norm)  # [B, H]
        map_emb = self.map_enc(map_feat)  # [B, L, H]

        # 2. Interaction Phase
        # -------------------------------------------------------------------------
        # [坐标系对齐 for PE]
        # 所有坐标必须在 Agent-Centric 下，并且数值范围接近 (e.g. [-1, 1])
        # -------------------------------------------------------------------------

        # A. Agent 位置
        # 在 Agent-Centric 下，当前时刻 (T=0) Agent 就在原点 (0,0)
        # hist_norm 是归一化后的，其最后一个点 (0,0) 也是 0 附近
        norm_agent_pos = torch.zeros(hist_norm.size(0), 2, device=hist_norm.device)

        # B. Map 位置
        # map_feat 是 Agent-Centric 的物理坐标 (范围约 ±50m)
        # 计算每条线段的物理中心
        raw_map_pos = map_feat[..., 0:2].mean(dim=2)  # [B, L, 2] (Physical)

        # 缩放: 除以 50.0 将物理坐标压缩到 [-1, 1] 区间，供 Fourier Feature 使用
        norm_map_pos = raw_map_pos / 50.0

        context_feat = self.interaction(agent_emb, map_emb, norm_agent_pos, norm_map_pos)

        # 3. Intention Injection
        intent_feat = self.action_embed(action_ids)  # [B, H]

        # 4. Decoding Phase (Physical Space)
        current_state_phys = raw_hist[:, -1, :]  # [B, 7]
        pred_traj = self.decoder(context_feat, intent_feat, current_state_phys)

        return pred_traj


# ==========================================
# Self-Testing Module
# ==========================================
if __name__ == "__main__":
    print("=== Testing Heterogeneous Predictor Model (Agent-Centric) ===")

    device = torch.device('cpu')
    model = HeterogeneousPredictor().to(device)

    # 构造 Dummy Data (Agent-Centric)
    B = 4
    # Hist: 归一化输入
    hist_norm = torch.randn(B, cfg.OBS_LEN, cfg.INPUT_DIM)

    # Raw Hist: 物理输入 (Agent-Centric)
    # T=0时刻 pos=(0,0), vel=(10, 0)
    raw_hist = torch.zeros(B, cfg.OBS_LEN, 7)
    raw_hist[:, :, 2] = 10.0  # vx = 10 m/s

    # Map: 物理输入 (Agent-Centric)
    map_feat = torch.randn(B, cfg.MAP_MAX_LINES, cfg.MAP_POINTS_PER_LINE, cfg.MAP_DIM)
    map_feat[..., :2] *= 20.0  # 模拟 20m 范围内的地图

    action_ids = torch.randint(0, cfg.NUM_ACTIONS, (B,))

    print(f"\nModel Architecture Initialized.")

    # Forward Pass
    try:
        pred = model(hist_norm, map_feat, action_ids, raw_hist)
        print(f"\nForward Pass Successful.")
        print(f"Output Pred: {pred.shape} (Expected: {B}, {cfg.PRED_LEN}, 4)")

        # 验证零初始化: 初始预测应接近匀速直线 (Constant Velocity)
        # CV Model: x = vt = 10 * t, y = 0
        expected_x_end = 10.0 * (cfg.PRED_LEN * 0.1)  # 20m
        pred_x_end = pred[0, -1, 0].item()

        print(f"CV Expected X: {expected_x_end:.2f}m")
        print(f"Model Pred X:  {pred_x_end:.2f}m")

        if abs(pred_x_end - expected_x_end) < 1.0:
            print("✅ Zero-Init Working: Model starts with valid physics.")
        else:
            print("❌ Zero-Init Failed: Output diverged.")

    except Exception as e:
        print(f"❌ Model Failed: {e}")
        import traceback

        traceback.print_exc()