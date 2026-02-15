import torch
import pickle
import numpy as np
import os
from torch.utils.data import Dataset
from config import cfg

class DeepAccidentDataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        
        # 1. 严格检查数据文件
        if not os.path.exists(cfg.PROCESSED_DATA_PATH):
            raise FileNotFoundError(
                f"[错误] 找不到处理后的数据: {cfg.PROCESSED_DATA_PATH}。\n"
                f"请先运行 data_process.py 进行真实数据预处理。"
            )
            
        print(f"[Dataset] 正在加载数据: {cfg.PROCESSED_DATA_PATH} ...")
        with open(cfg.PROCESSED_DATA_PATH, 'rb') as f:
            self.samples = pickle.load(f)
            
        with open(cfg.SCALER_PATH, 'rb') as f:
            self.scalers = pickle.load(f)
            
        # 转换为 Tensor 方便后续计算
        self.pos_mean = torch.FloatTensor(self.scalers['pos_mean'])
        self.pos_std  = torch.FloatTensor(self.scalers['pos_std'])
        self.vel_mean = torch.FloatTensor(self.scalers['vel_mean'])
        self.vel_std  = torch.FloatTensor(self.scalers['vel_std'])

    def _normalize(self, traj):
        """
        归一化轨迹 (输入网络前)
        """
        traj_norm = traj.clone()
        traj_norm[:, 0:2] = (traj[:, 0:2] - self.pos_mean) / self.pos_std
        traj_norm[:, 2:4] = (traj[:, 2:4] - self.vel_mean) / self.vel_std
        return traj_norm

    def _get_gt_action(self, hist, fut):
        """
        [Teacher Forcing 核心]
        根据未来的真实轨迹 (fut) 计算 Ground Truth Meta-Action。
        这确保了训练时 System 1 能够学习到正确的物理映射。
        """
        # 计算未来平均速度和当前速度
        curr_v = np.linalg.norm(hist[-1, 2:4])
        fut_v_avg = np.mean(np.linalg.norm(fut[:, 2:4], axis=1))
        
        # 计算横向位移 (未来终点相对于起点的角度)
        # 在 Agent-Centric 坐标下，起点为 (0,0)，朝向 +X
        end_x, end_y = fut[-1, 0], fut[-1, 1]
        heading_error = np.arctan2(end_y, end_x) # rad
        
        # 阈值定义 (根据物理规律调整)
        STOP_THRESH = 0.5   # m/s
        ACC_THRESH = 1.0    # m/s
        TURN_THRESH = 0.15  # rad (~8.5度)
        
        # 1. 判定纵向意图
        if curr_v < STOP_THRESH and fut_v_avg < STOP_THRESH:
            return 9 # Stop
            
        long_act = 0 # Keep
        if fut_v_avg - curr_v > ACC_THRESH: long_act = 1 # Acc
        elif fut_v_avg - curr_v < -ACC_THRESH: long_act = 2 # Dec
        
        # 2. 判定横向意图
        lat_base = 0 # Straight
        if heading_error > TURN_THRESH: lat_base = 3 # Left
        elif heading_error < -TURN_THRESH: lat_base = 6 # Right
        
        # 3. 组合 ID
        # Action Space mapping:
        # Straight: 0(Keep), 1(Acc), 2(Dec)
        # Left:     3(Keep), 4(Dec), 5(Acc)  <-- 注意 Config 定义顺序
        # Right:    6(Keep), 7(Dec), 8(Acc)
        
        final_id = 0
        if lat_base == 0: # Straight
            final_id = 0 + long_act
        elif lat_base == 3: # Left
            if long_act == 0: final_id = 3
            elif long_act == 2: final_id = 4 # Dec
            elif long_act == 1: final_id = 5 # Acc
        elif lat_base == 6: # Right
            if long_act == 0: final_id = 6
            elif long_act == 2: final_id = 7 # Dec
            elif long_act == 1: final_id = 8 # Acc
            
        return final_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        hist = torch.FloatTensor(sample['hist'])   # [30, 7]
        fut  = torch.FloatTensor(sample['future']) # [20, 4]
        map_feat = torch.FloatTensor(sample['map']) # [50, 20, 7]
        
        # 1. 归一化输入
        hist_norm = self._normalize(hist)
        
        # 2. 计算 GT 意图 (用于训练 System 1)
        gt_action = self._get_gt_action(sample['hist'], sample['future'])
        
        return {
            'hist_norm': hist_norm,
            'map_feat': map_feat,
            'fut_raw': fut,        # 物理真值 (用于 Loss)
            'hist_raw': hist,      # 物理历史 (用于 LLM Prompt)
            'gt_action': torch.tensor(gt_action, dtype=torch.long),
            'meta': str(sample['meta']) # 序列化元数据
        }
