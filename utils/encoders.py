import torch
import torch.nn as nn
import math
from config import cfg

class FourierEmbedding(nn.Module):
    """
    傅里叶位置编码 (Fourier Positional Embedding)
    将低维坐标映射到高维空间，增强 Transformer 对相对空间关系的感知能力。
    """
    def __init__(self, input_dim=2, hidden_dim=256, num_freqs=8):
        super().__init__()
        # 生成对数分布的频率频带
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        # 投影层: (input_dim * num_freqs * 2) -> hidden_dim
        self.out = nn.Linear(input_dim * num_freqs * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x: [..., input_dim] (归一化坐标)
        """
        # [..., D, 1] * [Freqs] -> [..., D, Freqs]
        x_exp = x.unsqueeze(-1) * self.freq_bands.to(x.device)
        x_exp = x_exp.view(*x.shape[:-1], -1)
        # sin/cos 变换 cat 在一起
        pe = torch.cat([torch.sin(x_exp), torch.cos(x_exp)], dim=-1)
        return self.norm(self.out(pe))

class AgentEncoder(nn.Module):
    """
    智能体历史轨迹编码器 (LSTM-based)
    捕捉非线性运动模式。
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.INPUT_DIM,
            hidden_size=cfg.HIDDEN_DIM,
            num_layers=2,  # 增加深度以捕获复杂动态
            batch_first=True,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(cfg.HIDDEN_DIM)

    def forward(self, x):
        # x: [B, T, D]
        self.lstm.flatten_parameters()
        _, (h, _) = self.lstm(x)
        # 取最后一层的最后时刻隐状态作为 Agent 特征
        return self.norm(h[-1]) # [B, H]

class MapEncoder(nn.Module):
    """
    地图矢量编码器 (VectorNet Style)
    处理矢量地图的无序点集。
    """
    def __init__(self):
        super().__init__()
        # Point-wise MLP 处理每个点的特征
        self.point_mlp = nn.Sequential(
            nn.Linear(cfg.MAP_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, cfg.HIDDEN_DIM),
            nn.LayerNorm(cfg.HIDDEN_DIM)
        )

    def forward(self, x):
        # x: [B, Lines, Points, D]
        B, L, P, D = x.shape
        x_flat = x.view(B * L, P, D)
        
        # 1. Point-wise Feature Extraction
        feat = self.point_mlp(x_flat) # [B*L, P, H]
        
        # 2. Aggregation (Max Pooling)
        # 聚合线段内所有点的信息，保证置换不变性
        feat, _ = torch.max(feat, dim=1) # [B*L, H]
        
        return feat.view(B, L, cfg.HIDDEN_DIM)

class SemanticRelativePositionEncoding(nn.Module):
    """
    语义增强相对位置编码 (备用模块)
    结合几何距离与语义属性（红绿灯、车道类型）的编码。
    """
    def __init__(self):
        super().__init__()
        self.geo_mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, cfg.HIDDEN_DIM)
        )
        # 语义属性嵌入 (假设 light_status有3种, lane_type有10种)
        self.light_emb = nn.Embedding(3, cfg.HIDDEN_DIM)
        self.lane_emb = nn.Embedding(10, cfg.HIDDEN_DIM)

    def forward(self, agent_i_state, agent_j_state, light_status, lane_type):
        # 几何状态计算
        dx = agent_j_state[..., 0] - agent_i_state[..., 0]
        dy = agent_j_state[..., 1] - agent_i_state[..., 1]
        rel_yaw = agent_j_state[..., 2] - agent_i_state[..., 2]
        
        geo_feat = torch.stack([
            dx, dy, torch.cos(rel_yaw), torch.sin(rel_yaw),
            agent_j_state[..., 3], agent_j_state[..., 4] # vx, vy
        ], dim=-1)
        
        # 特征融合
        geo_emb = self.geo_mlp(geo_feat)
        light_emb = self.light_emb(light_status)
        lane_emb = self.lane_emb(lane_type)
        
        return geo_emb + light_emb + lane_emb

class InteractionModule(nn.Module):
    """
    分层交互建模模块 (Hierarchical Interaction Module)
    1. 显式直接交互 (Direct Interaction)
    2. 隐式链式传播 (Chain Propagation)
    3. 安全优先聚合 (Safety-Priority Aggregation)
    """
    def __init__(self):
        super().__init__()
        # 2D 位置编码
        self.pe = FourierEmbedding(input_dim=2, hidden_dim=cfg.HIDDEN_DIM)
        
        # Layer 1: 显式直接交互 (基于距离的局部注意力)
        self.direct_interaction = nn.TransformerEncoderLayer(
            d_model=cfg.HIDDEN_DIM, 
            nhead=4, 
            dim_feedforward=1024,
            dropout=0.1, 
            batch_first=True, 
            norm_first=True
        )
        
        # Layer 2: 隐式链式传播 (全局感受野，捕捉蝴蝶效应)
        self.chain_propagation = nn.TransformerEncoderLayer(
            d_model=cfg.HIDDEN_DIM, 
            nhead=4, 
            dim_feedforward=1024,
            dropout=0.1, 
            batch_first=True, 
            norm_first=True
        )
        
        # Layer 3: Agent-Map Cross Attention (安全约束)
        self.agent_map_cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.HIDDEN_DIM, 
            num_heads=4, 
            batch_first=True
        )
        
        # 安全权重门控 (Safety Gate)
        # 对齐论文 4.2.3 节：对高风险地图元素（如停车线、红灯）赋予更高权重
        self.safety_weight_mlp = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, 1),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(cfg.HIDDEN_DIM)

    def forward(self, agent_emb, map_emb, agent_pos, map_pos):
        """
        Args:
            agent_emb: [B, H] 自车特征
            map_emb:   [B, L, H] 地图特征
            agent_pos: [B, 2] 自车归一化位置
            map_pos:   [B, L, 2] 地图线段中心归一化位置
        """
        # 1. 注入位置编码
        agent_pe = self.pe(agent_pos) # [B, H]
        agent_token = (agent_emb + agent_pe).unsqueeze(1) # [B, 1, H]
        
        map_pe = self.pe(map_pos) # [B, L, H]
        map_tokens = map_emb + map_pe # [B, L, H]
        
        # 2. 交互特征构建
        # 拼接 [Agent, Map...] 进行统一交互
        # 注意：为了适配分层结构，我们先让 Agent 和 Map 作为一个整体上下文进行交互
        tokens = torch.cat([agent_token, map_tokens], dim=1) # [B, 1+L, H]
        
        # 3. Layer 1: 直接交互
        direct_out = self.direct_interaction(tokens)
        
        # 4. Layer 2: 链式传播
        chain_out = self.chain_propagation(direct_out)
        
        # 提取更新后的 Agent 上下文
        agent_context = chain_out[:, 0, :] # [B, H]
        
        # 5. Layer 3: 安全优先聚合 (Safety-Weighted Map Aggregation)
        # 使用更新后的 Agent 特征去 Query 原始的 Map Tokens
        map_attn_out, _ = self.agent_map_cross_attn(
            query=agent_context.unsqueeze(1),
            key=map_tokens,
            value=map_tokens
        )
        
        # 计算安全权重 (Importance Gating)
        safety_weight = self.safety_weight_mlp(map_attn_out) # [B, 1, 1]
        
        # 加权融合: 越重要的地图特征（根据 Agent 状态判断），权重越大
        weighted_map_context = map_attn_out * safety_weight # [B, 1, H]
        
        # 6. 最终融合
        final_context = self.norm(agent_context + weighted_map_context.squeeze(1))
        
        return final_context
