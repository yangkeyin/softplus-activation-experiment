from .base_config import Config
import numpy as np

class X2AddY2(Config):
    # 继承基础配置，只修改不同的地方
    OUTPUT_DIR = "./results/target2D/x2_add_y2"
    MODEL_NAME = 'SimpleMLP2D'
    NUM_TRAIN_POINTS = 100
    NUM_TEST_POINTS = 100
    # 二维数据不进行频谱分析
    DATA_DIMENSION = 2
    DATA_RANGE = [[-2*np.pi, 2*np.pi], [-2*np.pi, 2*np.pi]]

    def target_function(self, x, y):
        return x**2 + y**2
  
# generate_data 改成 generate_data_2D

# plot_1d 改成 plot_all_2D