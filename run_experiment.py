import os
import pickle
import importlib
import torch
import torch.nn as nn
import numpy as np

# 从我们拆分好的模块中导入
from src.data_loader import generate_data, generate_inter_data, generate_outer_data, generate_2D_data
from src.models import SimpleMLP, SimpleMLP2D # 假设还有其他模型
from src.trainer import train_and_evaluate
from src.utils import get_fq_coef, rescale
from src.plothelper import plot_all

# 动态加载配置
# 这里可以用 argparse 让它更灵活，例如: python run_experiment.py --config=exp_sgd_lr_01
config_module = importlib.import_module("configs.target2D")
CFG = config_module.X2AddY2()


def main():
    # 1. 创建输出目录
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    # 2. 生成数据 (可以从配置中读取目标函数)
    X_train, y_train, X_test, y_test = generate_2D_data(
        CFG.DATA_RANGE, CFG.NUM_TRAIN_POINTS, CFG.NUM_TEST_POINTS, CFG.target_function, CFG.DEVICE
    )
    # 3. 计算目标函数的傅里叶系数
    # 归一化目标函数
    if CFG.DATA_DIMENSION == 1:
        normalized_target = lambda x: CFG.target_function(rescale(x, CFG.DATA_RANGE))
        true_coeffs = get_fq_coef(normalized_target)
    else:
        true_coeffs = None
    
    # ... 初始化 results 字典 ...
    results = {
        # 直接将测试数据存进去 (注意从Tensor转为Numpy array)
        'X_test': X_test.cpu().numpy(),
        'y_test': y_test.cpu().numpy(),
        'X_train': X_train.cpu().numpy(),
        'y_train': y_train.cpu().numpy(),
        'true_coeffs': true_coeffs,
        'fixed_params': {
            'beta': CFG.BETA_TO_RUN,
            'epochs': CFG.EPOCHS_LIST,
            'seed': CFG.SEED_LIST,
        },

        # 将原来的训练结果嵌套在一个新的键 'train_results' 中
        'train_results': {
            epochs: {
                    beta: {
                        seed: {'y_pred_std': None, 'y_pred': None, 'ann_coeffs': None} for seed in CFG.SEED_LIST
                    } for beta in CFG.BETA_TO_RUN
            } for epochs in CFG.EPOCHS_LIST
        }
    }

    for beta in CFG.BETA_TO_RUN:
        print(f"\n--- Training with beta={beta} ---")
        for seed in CFG.SEED_LIST:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # 3. 动态选择和实例化模型
            if CFG.MODEL_NAME == 'SimpleMLP':
                model = SimpleMLP(n_neurons=CFG.N_NEURONS, beta=beta).to(CFG.DEVICE)
            elif CFG.MODEL_NAME == 'SimpleMLP2D':
                model = SimpleMLP2D(n_neurons=CFG.N_NEURONS, beta=beta).to(CFG.DEVICE)
            # elif CFG.MODEL_NAME == 'AnotherModel':
            #     model = AnotherModel(...).to(CFG.DEVICE)
            else:
                raise ValueError(f"Unknown model name: {CFG.MODEL_NAME}")

            # 4. 动态选择优化器
            if CFG.OPTIMIZER_NAME == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)
            elif CFG.OPTIMIZER_NAME == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=CFG.LEARNING_RATE)
            else:
                raise ValueError(f"Unknown optimizer: {CFG.OPTIMIZER_NAME}")

            # 5. 调用训练函数 (将 optimizer 传入)
            train_and_evaluate(
                model=model,
                optimizer=optimizer,
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                config=CFG, # 传递整个配置对象
                beta=beta,
                seed=seed,
                results=results
            )

    # 6. 保存结果
    with open(os.path.join(CFG.OUTPUT_DIR, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    
    # visualize the results
    plot_all(CFG.OUTPUT_DIR, CFG.DATA_DIMENSION)


if __name__ == "__main__":
    main()