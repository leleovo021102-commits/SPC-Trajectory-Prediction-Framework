import torch
import os
import numpy as np
import json
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import cfg
from dataset import DeepAccidentDataset
from model import HeterogeneousPredictor
from llm_reasoning import SemanticProcessor, LLMClient


# 配置日志文件
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'inference_results.txt')
    # 清空旧日志
    with open(log_path, 'w') as f:
        f.write("=== Inference Log ===\n")
    return log_path


def log_to_file(path, msg):
    print(msg)
    with open(path, 'a') as f:
        f.write(msg + '\n')


def calculate_metrics(pred_traj, gt_traj):
    """
    计算物理空间的 ADE/FDE
    pred, gt: [B, T, 4] or [T, 4]
    """
    # 确保是 Tensor
    if not isinstance(pred_traj, torch.Tensor): pred_traj = torch.tensor(pred_traj)
    if not isinstance(gt_traj, torch.Tensor): gt_traj = torch.tensor(gt_traj)

    # 取前两维 (x, y)
    err = torch.norm(pred_traj[..., :2] - gt_traj[..., :2], dim=-1)

    ade = err.mean().item()
    fde = err[..., -1].mean().item()
    return ade, fde


def run_inference():
    # 1. 基础配置
    device = cfg.DEVICE
    if not hasattr(cfg, 'OUTPUT_DIR'): cfg.OUTPUT_DIR = './output'
    if not hasattr(cfg, 'CHECKPOINT_DIR'): cfg.CHECKPOINT_DIR = os.path.join(cfg.OUTPUT_DIR, 'checkpoints')

    log_path = setup_logging(cfg.OUTPUT_DIR)
    log_to_file(log_path, f"Running Inference on {device}...")

    # 2. 加载数据 (Validation Set)
    try:
        log_to_file(log_path, "Loading Dataset...")
        # 使用 mock=False 加载真实数据
        ds = DeepAccidentDataset(mode='val', mock=False)
        # 必须 batch_size=1，因为我们需要对每个样本单独生成 Prompt 并调用 LLM
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        log_to_file(log_path, f"Dataset Loaded. Total samples: {len(ds)}")
    except Exception as e:
        log_to_file(log_path, f"Error loading dataset: {e}")
        return

    # 3. 加载模型 (System 1)
    model = HeterogeneousPredictor().to(device)
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, 'best_model.pth')

    if os.path.exists(ckpt_path):
        log_to_file(log_path, f"Loading weights from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        log_to_file(log_path, f"Warning: Checkpoint not found at {ckpt_path}. Using random weights.")

    model.eval()

    # 4. 初始化 System 2 (LLM Client)
    # 建议使用 'deepseek' 或 'chatgpt'，如果没有 Key 会自动 Fallback 到规则
    llm_client = LLMClient(provider='deepseek')

    # 5. 统计指标容器
    metrics = {
        'total_samples': 0,
        'decision_correct': 0,
        'ade_llm': 0.0,  # 使用 LLM 意图的误差
        'fde_llm': 0.0,
        'ade_gt': 0.0,  # 使用 GT 意图的误差 (Upper Bound)
        'fde_gt': 0.0
    }

    results_buffer = []  # 用于保存详细结果到 .pkl

    # 6. 推理循环
    # 限制样本数以快速验证 (如需全量，请注释掉 if i >= MAX)
    MAX_SAMPLES = 50

    log_to_file(log_path, "\n--- Starting Bi-Level Inference Loop ---")

    for i, batch in enumerate(tqdm(loader)):
        if i >= MAX_SAMPLES: break

        # 搬运数据
        hist_norm = batch['hist_norm'].to(device)
        map_feat = batch['map_feat'].to(device)
        raw_hist = batch['raw_hist'].to(device)
        fut_phys = batch['fut_physical'].to(device)
        gt_action = batch['gt_action'].item()

        # ----------------------------------------------------
        # Step 1: System 2 Reasoning (LLM)
        # ----------------------------------------------------
        # 准备 Prompt
        raw_hist_np = raw_hist[0].cpu().numpy()

        # 处理 Meta (DataLoader 会把 dict 变成 list of values)
        meta = {}
        for k, v in batch['meta'].items():
            if isinstance(v, list) or isinstance(v, torch.Tensor):
                meta[k] = v[0] if len(v) > 0 else v
            else:
                meta[k] = v

        # 显式提取 Map 语义 (如果需要)
        # 这里 SemanticProcessor.generate_prompt 会内部调用 _analyze_map
        # 我们传入 map_feat (Tensor)
        map_feat_np = map_feat[0].cpu().numpy()
        prompt = SemanticProcessor.generate_prompt(raw_hist_np, map_feat_np, meta)

        # 调用 LLM
        pred_action_id, reasoning = llm_client.query(prompt)

        # ----------------------------------------------------
        # Step 2: System 1 Generation (Trajectory)
        # ----------------------------------------------------

        # Path A: Using LLM Predicted Action
        action_tensor_llm = torch.tensor([pred_action_id], device=device).long()
        with torch.no_grad():
            traj_llm = model(hist_norm, map_feat, action_tensor_llm, raw_hist)

        # Path B: Using GT Action (For Reference/Upper Bound)
        action_tensor_gt = torch.tensor([gt_action], device=device).long()
        with torch.no_grad():
            traj_gt = model(hist_norm, map_feat, action_tensor_gt, raw_hist)

        # ----------------------------------------------------
        # Step 3: Evaluation
        # ----------------------------------------------------
        ade_llm, fde_llm = calculate_metrics(traj_llm, fut_phys)
        ade_gt, fde_gt = calculate_metrics(traj_gt, fut_phys)

        metrics['total_samples'] += 1
        metrics['ade_llm'] += ade_llm
        metrics['fde_llm'] += fde_llm
        metrics['ade_gt'] += ade_gt
        metrics['fde_gt'] += fde_gt

        is_correct = (pred_action_id == gt_action)
        if is_correct:
            metrics['decision_correct'] += 1

        # ----------------------------------------------------
        # Logging & Saving
        # ----------------------------------------------------
        # 获取动作名称
        act_name_llm = cfg.ACTION_SPACE.get(pred_action_id, "Unknown")
        act_name_gt = cfg.ACTION_SPACE.get(gt_action, "Unknown")

        # 打印部分日志
        # 提取 Prompt 中的 Context 摘要
        try:
            ctx_summary = prompt.split('[Perception]')[1].split('[Rules]')[0].strip().replace('\n', '; ')
        except:
            ctx_summary = "Context Parse Error"

        log_msg = f"""
[Case {i + 1}]
Context:      {ctx_summary}
LLM Decision: ID {pred_action_id} ({act_name_llm})
GT Decision:  ID {gt_action} ({act_name_gt})
Reasoning:    {reasoning}
Metrics (LLM): ADE={ade_llm:.2f}m, FDE={fde_llm:.2f}m
Metrics (GT):  ADE={ade_gt:.2f}m, FDE={fde_gt:.2f}m
Match:        {'✅' if is_correct else '❌'}
------------------------------------------------"""
        log_to_file(log_path, log_msg)

        # 保存结构化结果以便后续分析/可视化
        results_buffer.append({
            'case_id': i,
            'hist': raw_hist_np,
            'fut': fut_phys[0].cpu().numpy(),
            'pred_traj': traj_llm[0].cpu().numpy(),
            'prompt': prompt,
            'reasoning': reasoning,
            'llm_action': pred_action_id,
            'gt_action': gt_action,
            'ade': ade_llm,
            'fde': fde_llm
        })

    # 7. 汇总报告
    total = metrics['total_samples']
    if total > 0:
        avg_ade_llm = metrics['ade_llm'] / total
        avg_fde_llm = metrics['fde_llm'] / total
        avg_ade_gt = metrics['ade_gt'] / total
        avg_fde_gt = metrics['fde_gt'] / total
        acc = metrics['decision_correct'] / total * 100

        summary = f"""
{'=' * 40}
FINAL INFERENCE REPORT
{'=' * 40}
Total Samples:      {total}
Decision Accuracy:  {acc:.2f}%
----------------------------------------
Trajectory Performance:
1. With LLM Action (Actual):
   ADE: {avg_ade_llm:.4f}m
   FDE: {avg_fde_llm:.4f}m

2. With GT Action (Upper Bound):
   ADE: {avg_ade_gt:.4f}m
   FDE: {avg_fde_gt:.4f}m

Diff (System 2 Cost): ADE +{avg_ade_llm - avg_ade_gt:.4f}m
{'=' * 40}
"""
        log_to_file(log_path, summary)

        # 保存 PKL 结果
        res_path = os.path.join(cfg.OUTPUT_DIR, 'inference_detailed_results.pkl')
        with open(res_path, 'wb') as f:
            pickle.dump(results_buffer, f)
        print(f"Detailed results saved to {res_path}")

    else:
        log_to_file(log_path, "No samples processed.")


if __name__ == "__main__":
    run_inference()