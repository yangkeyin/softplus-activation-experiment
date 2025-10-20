from .base_config import Config

class Outer1000(Config):
    # 继承基础配置，只修改不同的地方
    OUTPUT_DIR = "./results/outer/outer_1000"
    NUM_TRAIN_POINTS = 1000
    NUM_TEST_POINTS = 1000  