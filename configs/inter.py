from .base_config import Config

# 其他步骤
# 将generate data变成interp_data

# 10个训练点
class Inter10(Config):
    # 继承基础配置，只修改不同的地方
    OUTPUT_DIR = "./results/inter/inter_10"
    NUM_TRAIN_POINTS = 10
    NUM_TEST_POINTS = 1000

class Inter20(Config):
    # 继承基础配置，只修改不同的地方
    OUTPUT_DIR = "./results/inter/inter_20"
    NUM_TRAIN_POINTS = 20
    NUM_TEST_POINTS = 1000
    
class Inter50(Config):
    # 继承基础配置，只修改不同的地方
    OUTPUT_DIR = "./results/inter/inter_50"
    NUM_TRAIN_POINTS = 50
    NUM_TEST_POINTS = 1000  

class Inter100(Config):
    # 继承基础配置，只修改不同的地方
    OUTPUT_DIR = "./results/inter/inter_100"
    NUM_TRAIN_POINTS = 100
    NUM_TEST_POINTS = 1000  