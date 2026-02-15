import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import time
import numpy as np
from torch.cuda.amp import autocast, GradScaler

# 导入自定义模块
from config import cfg
from dataset import DeepAccidentDataset
from model import HeterogeneousPredictor
from loss_functions import SafetyLoss


# 简单的 Logging 工具
class Logger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, 'train_log.txt')
        with open(self.log_path, 'a') as f:
            f.write(f"\n{'=' * 20}\nTraining Session Start: {time.ctime()}\n{'=' * 20}\n")

    def log(self, msg):
        print(msg)
        with open(self.log_path, 'a') as f:
            f.write(msg + '\n')


def calculate_metrics(pred_traj, gt_traj):
    """
    计算 ADE 和 FDE (L2 距离)
    输入必须是物理空间（米）的数值
    """
    pred_pos = pred_traj[..., 0:2]
    gt_pos = gt_traj[..., 0:2]
    l2_dist = torch.norm(pred_pos - gt_pos, dim=-1)
    ade = l2_dist.mean(dim=-1).mean().item()
    fde = l2_dist[:, -1].mean().item()
    return ade, fde


def check_intention_compliance(pred_traj, action_ids):
    """
    [关键] 验证 System 1 是否听从 System 2 的指挥
    用于论文中的 "Controllability" 分析
    返回: Compliance Rate (0.0 - 1.0)
    """
    # 1. 检查 STOP 意图 (ID=9)
    # 规则: 如果意图是 Stop，末端速度应 < 0.5 m/s (或者位移非常小)
    is_stop = (action_ids == 9)
    if not is_stop.any():
        return -1.0  # 本批次无 Stop 样本

    # 计算末端速度 (vx, vy)
    # 假设 pred_traj [B, T, 4] -> [x, y, vx, vy]
    pred_vel = torch.norm(pred_traj[is_stop, -1, 2:4], dim=-1)

    # 判定合规: 速度小于阈值
    compliant_count = (pred_vel < 0.5).sum().item()
    total_stop = is_stop.sum().item()

    return compliant_count / total_stop


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_metrics = {'loss': 0.0, 'ade': 0.0, 'fde': 0.0}
    pbar = tqdm(loader, desc="Training", leave=False)

    for batch in pbar:
        # 数据搬运
        # Normalized input for encoders
        hist_norm = batch['hist_norm'].to(device)
        map_feat = batch['map_feat'].to(device)

        # Physical input for decoder initialization
        raw_hist = batch['raw_hist'].to(device)

        # Physical target for loss
        fut_phys = batch['fut_physical'].to(device)

        # Intent label
        gt_action = batch['gt_action'].to(device)

        optimizer.zero_grad()

        with autocast():
            # [核心] Teacher Forcing: 使用 GT Action 训练 System 1
            # 这确保 System 1 能够理解 Action Embedding 的物理含义
            pred_phys = model(hist_norm, map_feat, gt_action, raw_hist)
            loss, loss_dict = criterion(pred_phys, fut_phys, gt_action)

        # 混合精度反向传播
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        B = hist_norm.size(0)
        running_metrics['loss'] += loss.item() * B
        ade, fde = calculate_metrics(pred_phys, fut_phys)
        running_metrics['ade'] += ade * B
        running_metrics['fde'] += fde * B

        pbar.set_postfix({'loss': loss.item(), 'ade': ade})

    count = len(loader.dataset)
    return {k: v / count for k, v in running_metrics.items()}


def validate(model, loader, criterion, device):
    model.eval()
    running_metrics = {'loss': 0.0, 'ade': 0.0, 'fde': 0.0, 'compliance': 0.0}
    compliance_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            hist_norm = batch['hist_norm'].to(device)
            map_feat = batch['map_feat'].to(device)
            raw_hist = batch['raw_hist'].to(device)
            fut_phys = batch['fut_physical'].to(device)
            gt_action = batch['gt_action'].to(device)

            pred_phys = model(hist_norm, map_feat, gt_action, raw_hist)
            loss, _ = criterion(pred_phys, fut_phys, gt_action)

            # 基础指标
            B = hist_norm.size(0)
            running_metrics['loss'] += loss.item() * B
            ade, fde = calculate_metrics(pred_phys, fut_phys)
            running_metrics['ade'] += ade * B
            running_metrics['fde'] += fde * B

            # [关键] 验证 System 1 是否 "听话" (Control Controllability)
            comp_rate = check_intention_compliance(pred_phys, gt_action)
            if comp_rate >= 0:
                running_metrics['compliance'] += comp_rate
                compliance_batches += 1

    count = len(loader.dataset)
    avg_metrics = {k: v / count for k, v in running_metrics.items()}

    # 修正 Compliance 的平均值计算
    if compliance_batches > 0:
        avg_metrics['compliance'] = running_metrics['compliance'] / compliance_batches
    else:
        avg_metrics['compliance'] = 0.0

    return avg_metrics


def main():
    # 1. 路径与配置
    if not hasattr(cfg, 'OUTPUT_DIR'): cfg.OUTPUT_DIR = './output'
    if not hasattr(cfg, 'CHECKPOINT_DIR'): cfg.CHECKPOINT_DIR = os.path.join(cfg.OUTPUT_DIR, 'checkpoints')

    try:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    except PermissionError:
        print("Permission Error: Please check config.py paths or manually create output dir.")
        # Fallback to local dir if absolute path fails
        cfg.OUTPUT_DIR = './output'
        cfg.CHECKPOINT_DIR = './output/checkpoints'
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    logger = Logger(cfg.OUTPUT_DIR)
    device = cfg.DEVICE

    # 2. 数据加载
    try:
        logger.log(f"Loading data from {cfg.PROCESSED_DATA_PATH}")
        full_dataset = DeepAccidentDataset(mode='train', mock=False)
        train_len = int(0.8 * len(full_dataset))
        train_set, val_set = random_split(full_dataset, [train_len, len(full_dataset) - train_len])

        train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.NUM_WORKERS, pin_memory=True)
        logger.log(f"Dataset Loaded. Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    except Exception as e:
        logger.log(f"Error loading real data: {e}")
        logger.log("Fallback to Mock Data for debugging...")
        train_set = DeepAccidentDataset(mock=True)
        val_set = DeepAccidentDataset(mock=True)
        train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE)
        val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE)

    # 3. 模型构建
    model = HeterogeneousPredictor().to(device)
    criterion = SafetyLoss().to(device)

    # 稍微增大 LR 以适应物理空间较大的 Loss 值
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    scaler = GradScaler()

    # 4. 训练循环
    logger.log(f"\nStart Training on {device} (Physical Space with Residual Decoding)...")
    best_val_ade = float('inf')

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)

        # Val
        val_metrics = validate(model, val_loader, criterion, device)

        # LR Update
        scheduler.step(val_metrics['ade'])

        duration = time.time() - t0

        # Logging
        # 这里的 Compliance Rate 是为了向审稿人证明：我们的 System 1 确实受到了 System 2 的约束
        logger.log(f"Epoch {epoch + 1}/{cfg.EPOCHS} | T: {duration:.1f}s")
        logger.log(f"  Train: Loss={train_metrics['loss']:.2f} ADE={train_metrics['ade']:.2f}m")
        logger.log(
            f"  Val:   Loss={val_metrics['loss']:.2f} ADE={val_metrics['ade']:.2f}m FDE={val_metrics['fde']:.2f}m")
        logger.log(f"  >> Intent Compliance (Stop Accuracy): {val_metrics['compliance'] * 100:.1f}%")

        # Save Best
        if val_metrics['ade'] < best_val_ade:
            best_val_ade = val_metrics['ade']
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, 'best_model.pth'))
            logger.log("  --> Best Model Saved")

        # Regular Checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f'epoch_{epoch + 1}.pth'))

    logger.log("\nTraining Finished.")


if __name__ == "__main__":
    main()