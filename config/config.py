import torch
import os


class Config:
    # ----------------------------------
    # 1. 路径配置 (硬编码适配云端环境)
    # ----------------------------------
    USER_ROOT = '/home/carla_user'

    # 原始数据路径
    RAW_DATA_ROOT = './data'

    # 输出路径
    OUTPUT_DIR = os.path.join(USER_ROOT, 'output')
    PROCESSED_DATA_PATH = os.path.join(USER_ROOT, 'processed_data_enhanced.pkl')
    SCALER_PATH = os.path.join(USER_ROOT, 'scalers.pkl')
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

    # ----------------------------------
    # 2. 数据维度
    # ----------------------------------
    INPUT_DIM = 7  # Agent: (x, y, vx, vy, ax, ay, yaw)

    # [关键修正] 适配真实数据的 7 维地图特征
    # 通常包含: x, y, z, type, heading, speed_limit, lane_id/validity
    MAP_DIM = 7
    HIDDEN_DIM = 256

    OBS_LEN = 30  # 3s
    PRED_LEN = 20  # 2s

    # 地图处理参数
    MAP_MAX_LINES = 50
    MAP_POINTS_PER_LINE = 20

    # ----------------------------------
    # 3. 动作空间
    # ----------------------------------
    ACTION_SPACE = {
        0: "Straight_Keep", 1: "Straight_Acc", 2: "Straight_Dec",
        3: "Left_Keep", 4: "Left_Dec", 5: "Left_Acc",
        6: "Right_Keep", 7: "Right_Dec", 8: "Right_Acc",
        9: "Stop"
    }
    NUM_ACTIONS = 10

    # ----------------------------------
    # 4. 训练参数
    # ----------------------------------
    BATCH_SIZE = 512  # 增大 Batch Size 稳定梯度
    LEARNING_RATE = 1e-3
    EPOCHS = 200
    NUM_WORKERS = 16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cfg = Config()

if __name__ == "__main__":
    try:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        print(f"✅ Directories ready at {cfg.USER_ROOT}")
    except PermissionError:
        print(f"❌ Permission Denied at {cfg.USER_ROOT}")