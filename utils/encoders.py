import torch
import torch.nn as nn
from config import cfg
import math


class FourierEmbedding(nn.Module):
    """
    Fourier Positional Embedding
    Maps low-dimensional coordinates (x, y) to high-dimensional feature space,
    helping Transformer perceive relative spatial relationships.
    """

    def __init__(self, input_dim=2, hidden_dim=64, num_freqs=8):
        super().__init__()
        # Generate logarithmically spaced frequencies
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        # Projection layer: (input_dim * num_freqs * 2) -> hidden_dim
        self.out = nn.Linear(input_dim * num_freqs * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x: [..., input_dim] (normalized coordinates, usually between -1 and 1)
        """
        # [..., D, 1] * [Freqs] -> [..., D, Freqs]
        x_exp = x.unsqueeze(-1) * self.freq_bands.to(x.device)
        x_exp = x_exp.view(*x.shape[:-1], -1)
        # sin/cos transformation
        pe = torch.cat([torch.sin(x_exp), torch.cos(x_exp)], dim=-1)
        return self.norm(self.out(pe))


class AgentEncoder(nn.Module):
    """
    LSTM-based historical trajectory encoder
    Input: Normalized historical trajectory
    """

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(cfg.INPUT_DIM, cfg.HIDDEN_DIM, batch_first=True)

    def forward(self, x):
        # x: [B, T, D]
        # Use the hidden state of the last time step as Agent feature
        _, (h, _) = self.lstm(x)
        return h[-1]  # [B, H]


class MapEncoder(nn.Module):
    """
    PointNet-based map encoder (Adapted for 7-dimensional features)
    """

    def __init__(self):
        super().__init__()
        # Adapt to MAP_DIM (7) in configuration
        self.point_mlp = nn.Sequential(
            nn.Linear(cfg.MAP_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, cfg.HIDDEN_DIM)
        )

    def forward(self, x):
        # x: [B, L, P, D]
        # 1. Point-wise feature extraction
        feat = self.point_mlp(x)  # [B, L, P, H]

        # 2. Aggregate line segment features (Max Pooling over Points)
        # Ensures features are insensitive to point order
        feat, _ = torch.max(feat, dim=2)  # [B, L, H]

        return feat


class InteractionModule(nn.Module):
    """
    Multimodal Interaction Module (Transformer)
    Fuses Agent historical features and Map environmental features
    """

    def __init__(self):
        super().__init__()
        # 2D Positional Encoder (x, y)
        self.pe = FourierEmbedding(input_dim=2, hidden_dim=cfg.HIDDEN_DIM)

        # Transformer Encoder
        # Use Pre-Norm (norm_first=True) for better training stability
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.HIDDEN_DIM,
                nhead=4,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True,
                norm_first=True
            ),
            num_layers=2
        )

    def forward(self, agent_emb, map_emb, agent_pos, map_pos):
        """
        Args:
            agent_emb: [B, H] Agent semantic features
            map_emb:   [B, L, H] Map semantic features
            agent_pos: [B, 2] Agent normalized position
            map_pos:   [B, L, 2] Map line segment normalized center position
        """
        # 1. Inject Positional Encoding
        agent_pe = self.pe(agent_pos)  # [B, H]
        agent_token = (agent_emb + agent_pe).unsqueeze(1)  # [B, 1, H]

        map_pe = self.pe(map_pos)  # [B, L, H]
        map_tokens = map_emb + map_pe  # [B, L, H]

        # 2. Concatenate Tokens
        # [Agent, Map_1, Map_2, ..., Map_L]
        tokens = torch.cat([agent_token, map_tokens], dim=1)  # [B, 1+L, H]

        # 3. Global Interaction
        out = self.transformer(tokens)

        # 4. Return updated Agent feature (0th token)
        return out[:, 0, :]