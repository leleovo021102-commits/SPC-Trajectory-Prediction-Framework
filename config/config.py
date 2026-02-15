import torch
import os

class Config:
    # ----------------------------------
    # 1. 路径配置 (兼容本地/云端环境)
    # ----------------------------------
    USER_ROOT = os.path.expanduser('~')
    # 数据集根目录（DeepAccident官方数据集解压路径）
    RAW_DATA_ROOT = './DeepAccident'
    # 输出路径
    OUTPUT_DIR = os.path.join(USER_ROOT, 'spc_trajectory_output')
    PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, 'processed_deepaccident.pkl')
    SCALER_PATH = os.path.join(OUTPUT_DIR, 'scalers.pkl')
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

    # ----------------------------------
    # 2. 数据维度 
    # ----------------------------------
    # Agent状态维度: (x, y, vx, vy, ax, ay, yaw)
    INPUT_DIM = 7
    # 地图特征维度: (x, y, z, lane_type, heading, speed_limit, traffic_light)
    MAP_DIM = 7
    HIDDEN_DIM = 256  #统一隐藏层维度
    OBS_LEN = 20
    PRED_LEN = 30
    # 地图处理参数
    MAP_MAX_LINES = 50
    MAP_POINTS_PER_LINE = 20

    # ----------------------------------
    # 3. 元动作空间 - 
    # 横向(左/直/右) × 纵向(加速/保持/减速/停止)，共10个正交元动作
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
    BATCH_SIZE = 256
    LEARNING_RATE = 3e-4  
    EPOCHS = 200
    NUM_WORKERS = 16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------
    # 5. 损失函数权重 
    # ----------------------------------
    LOSS_WEIGHTS = {
        "reg": 1.0,
        "intent": 0.5,
        "phy": 0.1,
        "scene": 0.8
    }

    # ----------------------------------
    # 6. LLM推理配置
    # ----------------------------------
    LLM_TEMPERATURE = 0.0  # 确定性推理
    LLM_MAX_TOKENS = 512
    # API密钥从环境变量读取
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

cfg = Config()

# 初始化目录
if __name__ == "__main__":
    for dir_path in [cfg.OUTPUT_DIR, cfg.CHECKPOINT_DIR, cfg.LOG_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    print(f"✅ 目录初始化完成，根路径: {cfg.OUTPUT_DIR}")
