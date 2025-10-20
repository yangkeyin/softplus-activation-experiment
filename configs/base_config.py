from re import X
import torch
import numpy as np

class Config:
    # --- 设备配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 数据集参数 ---
    DATA_RANGE = [-2 * np.pi, 2 * np.pi]
    NUM_POINTS = 1000
    TEST_SIZE = 0.2

    DATA_DIMENSION = 1

    # --- 模型与训练参数 ---
    MODEL_NAME = 'SimpleMLP' # 用于选择模型
    N_NEURONS = 64
    BETA_TO_RUN = [0.5, 1, 2, 4, 8, 10, 20, 50]
    EPOCHS_LIST = [100, 500, 1000, 2000]

    # --- 优化器参数 ---
    OPTIMIZER_NAME = 'Adam'  # Adam, SGD, etc.
    LEARNING_RATE = 0.001

    # --- 输出与种子 ---
    OUTPUT_DIR = "./results/default_experiment"
    SEED_LIST = [42, 38, 100]

    def target_function(self, x):
        return np.sin(x)