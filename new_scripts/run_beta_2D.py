import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import sys
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从utils.py中只导入set_seed函数
# 注意：这里假设utils.py中已经有set_seed函数，否则需要在本文件中实现
def set_seed(seed_value):
    """设置所有需要随机种子的库的种子。"""
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    # Python的内置随机库
    import random
    random.seed(seed_value)

# 配置参数
BETA = [0.25, 0.5, 1.0, 4.0, 8.0, 16.0]
TARGET_FUNC_2D = lambda x, y: x**2 + y**2
DATA_RANGE = [-2 * np.pi, 2 * np.pi]
EPOCHS = 10000
SEEDS = [100, 200, 300]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASELINE_EPOCH = 10000  # 用于保存基线模型的epoch

# 输出目录配置 - 修改为符合微调脚本要求的结构
OUTPUT_DIR = "figures/beta_2D_x2ADDy2_SGD_momentum0.9_lr0.002"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 创建模型保存目录
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class FNNModel2D(nn.Module):
    def __init__(self, n, beta):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, n),  # 接受 (x, y) 两个输入
            nn.Softplus(beta=beta),
            nn.Linear(n, 1)
        )
    def forward(self, x):
        return self.layers(x)


def generate_data_2D(data_range, train_points_per_dim, test_points_per_dim, device):
    """
    生成2D网格数据
    """
    # 1. 训练数据 (9x9 网格)
    x_ticks = np.linspace(data_range[0], data_range[1], train_points_per_dim)
    y_ticks = np.linspace(data_range[0], data_range[1], train_points_per_dim)
    xx_train, yy_train = np.meshgrid(x_ticks, y_ticks)
    X_train_np = np.vstack([xx_train.ravel(), yy_train.ravel()]).T
    # 使用全局定义的目标函数
    y_train_np = TARGET_FUNC_2D(X_train_np[:, 0], X_train_np[:, 1])
    
    # 2. 测试数据 (50x50 网格)
    x_ticks_test = np.linspace(data_range[0], data_range[1], test_points_per_dim)
    y_ticks_test = np.linspace(data_range[0], data_range[1], test_points_per_dim)
    xx_test, yy_test = np.meshgrid(x_ticks_test, y_ticks_test)
    X_test_np = np.vstack([xx_test.ravel(), yy_test.ravel()]).T
    # 使用全局定义的目标函数
    y_test_np = TARGET_FUNC_2D(X_test_np[:, 0], X_test_np[:, 1])

    # 3. 转换为张量
    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).reshape(-1, 1).to(device)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test_np, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # 返回 (X_test, y_test) 和 (xx_test, yy_test) 等以便绘图
    return X_train, y_train, X_test, y_test, xx_train, yy_train, xx_test, yy_test


def plot_each_epoch_2D(results, xx_train, yy_train, y_train, xx_test, yy_test, y_test, beta, output_dir):
    seeds = list(results.keys())
    epochs = list(results[seeds[0]].keys())
    for epoch in epochs:
        # 可视化训练结果
        fig, ax = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f"Softplus Activation with Beta={beta} Epoch {epoch}")
        
        # 计算每个seed下所有指标的平均值
        avg_train_rms = np.mean([results[seed][epoch]["train_rms"] for seed in seeds])
        avg_test_rms = np.mean([results[seed][epoch]["test_rms"] for seed in seeds])
        std_train_rms = np.std([results[seed][epoch]["train_rms"] for seed in seeds])
        std_test_rms = np.std([results[seed][epoch]["test_rms"] for seed in seeds])
        
        # 重塑预测结果以适应网格形状
        train_shape = xx_train.shape
        test_shape = xx_test.shape
        
        # 计算平均预测结果
        avg_y_pred = np.mean([results[seed][epoch]["y_pred"].reshape(train_shape) for seed in seeds], axis=0)
        avg_y_pred_test = np.mean([results[seed][epoch]["y_pred_test"].reshape(test_shape) for seed in seeds], axis=0)
        
        # 重塑真实值以适应网格形状
        y_train_grid = y_train.reshape(train_shape)
        y_test_grid = y_test.reshape(test_shape)
        
        # 行 0: 训练集 (9x9)
        # [0, 0]: 真实训练 Z
        im0 = ax[0, 0].imshow(y_train_grid, extent=[DATA_RANGE[0], DATA_RANGE[1], DATA_RANGE[0], DATA_RANGE[1]], 
                           origin='lower', cmap='viridis')
        ax[0, 0].set_title(f"True Training Z (9x9 grid)")
        plt.colorbar(im0, ax=ax[0, 0])
        
        # [0, 1]: 平均预测 Z (训练集)
        im1 = ax[0, 1].imshow(avg_y_pred, extent=[DATA_RANGE[0], DATA_RANGE[1], DATA_RANGE[0], DATA_RANGE[1]], 
                           origin='lower', cmap='viridis')
        ax[0, 1].set_title(f"Average Predicted Z (Training Set)\nAVG RMS: {avg_train_rms:.6f}, STD RMS: {std_train_rms:.6f}")
        plt.colorbar(im1, ax=ax[0, 1])
        
        # [0, 2]: 平均训练误差
        avg_train_error = avg_y_pred - y_train_grid
        im2 = ax[0, 2].imshow(avg_train_error, extent=[DATA_RANGE[0], DATA_RANGE[1], DATA_RANGE[0], DATA_RANGE[1]], 
                           origin='lower', cmap='RdBu')
        ax[0, 2].set_title(f"Average Training Error")
        plt.colorbar(im2, ax=ax[0, 2])
        
        # 行 1: 测试集 (50x50)
        # [1, 0]: 真实测试 Z
        im3 = ax[1, 0].imshow(y_test_grid, extent=[DATA_RANGE[0], DATA_RANGE[1], DATA_RANGE[0], DATA_RANGE[1]], 
                           origin='lower', cmap='viridis')
        ax[1, 0].set_title(f"True Test Z (50x50 grid)")
        plt.colorbar(im3, ax=ax[1, 0])
        
        # [1, 1]: 平均预测 Z (测试集)
        im4 = ax[1, 1].imshow(avg_y_pred_test, extent=[DATA_RANGE[0], DATA_RANGE[1], DATA_RANGE[0], DATA_RANGE[1]], 
                           origin='lower', cmap='viridis')
        ax[1, 1].set_title(f"Average Predicted Z (Test Set)\nAVG RMS: {avg_test_rms:.6f}, STD RMS: {std_test_rms:.6f}")
        plt.colorbar(im4, ax=ax[1, 1])
        
        # [1, 2]: 平均测试误差
        avg_test_error = avg_y_pred_test - y_test_grid
        im5 = ax[1, 2].imshow(avg_test_error, extent=[DATA_RANGE[0], DATA_RANGE[1], DATA_RANGE[0], DATA_RANGE[1]], 
                           origin='lower', cmap='RdBu')
        ax[1, 2].set_title(f"Average Test Error")
        plt.colorbar(im5, ax=ax[1, 2])
        
        # 添加公共的x轴和y轴标签
        for i in range(2):
            for j in range(3):
                ax[i, j].set_xlabel('x')
                ax[i, j].set_ylabel('y')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        fig.savefig(f"{output_dir}/beta_{beta}_epoch_{epoch}.png")
        plt.close(fig)  # 关闭当前figure，释放内存
        print(f"Training completed. Saved figures to {output_dir}/beta_{beta}_epoch_{epoch}.png")


def main():
    # 准备数据 - 调用generate_data_2D函数
    X_train, y_train, X_test, y_test, xx_train, yy_train, xx_test, yy_test = generate_data_2D(DATA_RANGE, 9, 50, DEVICE)
    
    # 初始化基线结果字典
    results_base = {
        'X_train': X_train.cpu().numpy(),
        'y_train': y_train.cpu().numpy(),
        'X_test': X_test.cpu().numpy(),
        'y_test': y_test.cpu().numpy(),
        'xx_train': xx_train,
        'yy_train': yy_train,
        'xx_test': xx_test,
        'yy_test': yy_test
    }

    # 为每个beta都定义一个颜色
    beta_colors = plt.cm.tab10(np.linspace(0, 1, len(BETA)))
    color_map = {beta: beta_colors[i] for i, beta in enumerate(BETA)}

    # 初始化存储训练结果的字典
    rms_data = {}
    # 初始化基线模型结果字典 - 按[beta][seed][epoch]组织
    baseline_metrics = {beta: {} for beta in BETA}
    
    # 为每一个beta值训练一个模型
    for beta in BETA:
        print(f"Training beta={beta}")
        results = {}
        rms_data[beta] = {}
        baseline_metrics[beta] = {}
        
        # 定义输出目录
        output_dir = f"{OUTPUT_DIR}/beta_{beta}/"
        os.makedirs(output_dir, exist_ok=True)
        
        for seed in SEEDS:
            print(f"Training beta={beta}, seed={seed}")
            set_seed(seed)
            results[seed] = {}
            rms_data[beta][seed] = {}
            baseline_metrics[beta][seed] = {}

            # 搭建模型 - 使用FNNModel2D
            model = FNNModel2D(n=100, beta=beta)
            model.to(DEVICE)

            # 训练模型
            criterion = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
            num_epochs = EPOCHS
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                y_pred = model(X_train)
                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()

                if (epoch+1) % 100 == 0:
                    rms_data[beta][seed][epoch + 1] = {}
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

                    # 测试模型
                    model.eval()
                    with torch.no_grad():
                        y_pred_test = model(X_test)
                        test_loss = criterion(y_pred_test, y_test)
                    
                    # 计算所有的误差（只计算train_rms和test_rms，移除spectrum_error）
                    train_rms = np.sqrt(loss.item())
                    test_rms = np.sqrt(test_loss.item())

                    # 将要可视化的数据存储到results中
                    results[seed][epoch + 1] = {}
                    results[seed][epoch + 1]["y_pred"] = y_pred.cpu().detach().numpy().flatten()
                    results[seed][epoch + 1]["y_pred_test"] = y_pred_test.cpu().detach().numpy().flatten()
                    results[seed][epoch + 1]["train_rms"] = train_rms
                    results[seed][epoch + 1]["test_rms"] = test_rms
                    
                    rms_data[beta][seed][epoch + 1]["train_rms"] = train_rms
                    rms_data[beta][seed][epoch + 1]["test_rms"] = test_rms
                    
                    # 保存基线指标
                    baseline_metrics[beta][seed][epoch + 1] = {
                        "test_rms_base": test_rms,
                        "y_pred_test_base": y_pred_test.cpu().detach().numpy().flatten(),
                        "y_pred_train_base": y_pred.cpu().detach().numpy().flatten()
                    }
                    
                    # 保存模型 - 如果达到基线epoch
                    if epoch + 1 == BASELINE_EPOCH:
                        model_path = os.path.join(MODELS_DIR, f"model_beta_{beta}_seed_{seed}_epoch_{BASELINE_EPOCH}.pth")
                        torch.save(model.state_dict(), model_path)
                        print(f"Saved model to {model_path}")

        # 可视化训练结果 - 调用plot_each_epoch_2D函数
        plot_each_epoch_2D(results, xx_train, yy_train, y_train.cpu().numpy().flatten(), 
                          xx_test, yy_test, y_test.cpu().numpy().flatten(), beta, output_dir)
    
    # 利用rms_list绘制beta与rms的关系图
    fig_seed, ax_seed = plt.subplots(2, 1, figsize=(12, 12), sharex=True) # 共享x轴，改为2x1布局
    
    # 计算每个beta值下每个epoch的平均rms值
    avg_rms_data = {}
    epoch_rms_data = {}
    for beta in BETA:
        avg_rms_data[beta] = {}
        epoch_rms_data[beta] = {}
        for epoch in rms_data[beta][SEEDS[0]]:
            avg_rms_data[beta][epoch] = {}
            avg_rms_data[beta][epoch]["train_rms"] = np.mean([rms_data[beta][seed][epoch]["train_rms"] for seed in SEEDS])
            avg_rms_data[beta][epoch]["test_rms"] = np.mean([rms_data[beta][seed][epoch]["test_rms"] for seed in SEEDS])
        for seed in SEEDS:
            epoch_rms_data[beta][seed] = {}
            epoch_rms_data[beta][seed]["train_rms"] = np.array([rms_data[beta][seed][epoch]["train_rms"] for epoch in avg_rms_data[beta]])
            epoch_rms_data[beta][seed]["test_rms"] = np.array([rms_data[beta][seed][epoch]["test_rms"] for epoch in avg_rms_data[beta]])

    for beta in BETA:
        epochs_recorded = sorted(avg_rms_data[beta].keys())
        rms_value = np.array([avg_rms_data[beta][epoch]['train_rms'] for epoch in epochs_recorded])
        test_rms_value = np.array([avg_rms_data[beta][epoch]['test_rms'] for epoch in epochs_recorded])

        ax_seed[0].semilogy(epochs_recorded, rms_value, "-o", label=f"Beta={beta}", color=color_map[beta])
        ax_seed[1].semilogy(epochs_recorded, test_rms_value, "-o", label=f"Beta={beta}", color=color_map[beta])

    ax_seed[0].set_title(f"Train RMS vs Epoch")
    ax_seed[0].set_xlabel("Epoch")
    ax_seed[0].set_ylabel("Train RMS")
    ax_seed[0].legend()
    ax_seed[0].grid(which='both', linestyle='--', linewidth=0.5)
    
    ax_seed[1].set_title(f"Test RMS vs Epoch")
    ax_seed[1].set_xlabel("Epoch")
    ax_seed[1].set_ylabel("Test RMS")
    ax_seed[1].legend()
    ax_seed[1].grid(which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    fig_seed.savefig(f"{OUTPUT_DIR}/beta_rms_seed.png")
    plt.close(fig_seed)  # 关闭当前figure，释放内存
    print(f"RMS plot saved to {OUTPUT_DIR}/beta_rms_seed.png")
    
    # 保存基线结果
    results_base['BETA'] = BETA
    results_base['SEEDS'] = SEEDS
    results_base['BASELINE_EPOCH'] = BASELINE_EPOCH
    results_base['metrics'] = baseline_metrics
    
    # 保存基线结果到pkl文件
    baseline_file = os.path.join(OUTPUT_DIR, "results_base_2D.pkl")
    with open(baseline_file, 'wb') as f:
        pickle.dump(results_base, f)
    print(f"Saved baseline results to {baseline_file}")


if __name__ == "__main__":
    main()