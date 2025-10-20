from .base_config import Config

class ExperimentConfig(Config):
    # 继承基础配置，只修改不同的地方
    OPTIMIZER_NAME = 'Adam'
    LEARNING_RATE = 0.001
    OUTPUT_DIR = "./results/adam_lr_001"
