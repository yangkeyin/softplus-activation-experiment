import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import sys
import pickle
import datetime
from scipy.interpolate import interp1d
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import rescale, get_fq_coef

# 配置参数
BETA = [1.0, 4.0, 8.0, 16.0]
TARGET_FUNC = lambda x: torch.sin(x)
DATA_RANGE = [-2 * np.pi, 2 * np.pi]
EPOCHS = 10000
SEEDS = [100, 200, 300, 400, 500]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASELINE_EPOCH = 10000  # 用于保存基线模型的epoch
INIT_STD = None # 初始化标准差

# 输出目录配置 - 修改为符合微调脚本要求的结构
date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"figures/beta_base_{date_time}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 创建模型保存目录
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class FNNModel(nn.Module):
    def __init__(self, n, beta):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, n),
            nn.Softplus(beta=beta),
            nn.Linear(n, 1)
        )
        if INIT_STD is not None:
            self.layers.apply(lambda m: self._init_weights(m, INIT_STD))    
    def forward(self, x):
        return self.layers(x)
    
    def _init_weights(self, m, init_std):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=init_std)
            nn.init.constant_(m.bias, 0.0)
        
def set_seed(seed_value):
    """设置所有需要随机种子的库的种子。"""
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    # Python的内置随机库
    import random
    random.seed(seed_value)


def plot_each_epoch(results, x_train, y_train, x_test, y_test, true_coef, beta, output_dir, std= 0.5):
    seeds = list(results.keys())
    epochs = list(results[seeds[0]].keys())
    for epoch in epochs:
        # 可视化训练结果
        fig, ax = plt.subplots(3,2, figsize=(20, 15))
        fig.suptitle(f"Softplus (Beta={beta}, Std={std}) Epoch {epoch+1}")
        # 计算每个seed下所有指标的平均值
        avg_train_rms = np.mean([results[seed][epoch]["train_rms"] for seed in seeds])
        avg_test_rms = np.mean([results[seed][epoch]["test_rms"] for seed in seeds])
        avg_spectrum_error = np.mean([results[seed][epoch]["spectrum_error"] for seed in seeds])
        std_train_rms = np.std([results[seed][epoch]["train_rms"] for seed in seeds])
        std_test_rms = np.std([results[seed][epoch]["test_rms"] for seed in seeds])
        std_spectrum_error = np.std([results[seed][epoch]["spectrum_error"] for seed in seeds])
        avg_y_pred = np.mean([results[seed][epoch]["y_pred"] for seed in seeds], axis=0)
        avg_pred_coef = np.mean([results[seed][epoch]["pred_coef"] for seed in seeds], axis=0)
        avg_y_pred_test = np.mean([results[seed][epoch]["y_pred_test"] for seed in seeds], axis=0)

        # 先绘制平均结果
        ax[0][0].set_title(f"Fit train function with {x_train.shape[0]} points")
        ax[0][0].plot(x_train, y_train, 'r--o', label=f"True train sin(x)")
        ax[0][0].plot(x_train, avg_y_pred, 'b-o', label=f"Avg train sin(x)")

        ax[0][1].set_title(f"Fit train function loss, AVG RMS: {avg_train_rms:.6f}, STD RMS: {std_train_rms:.6f}")
        ax[0][1].plot(x_train, avg_y_pred - y_train, 'b-o', label=f"Avg train Loss")
        

        ax[1][0].set_title(f"Fit test function with {x_test.shape[0]} points")
        ax[1][0].plot(x_train, y_train, 'k-', label="True train sin(x)")    
        ax[1][0].plot(x_test, y_test, 'r--o', label="True test sin(x)")
        ax[1][0].plot(x_test, avg_y_pred_test, 'b-o', label=f"Avg test sin(x)")

        ax[1][1].set_title(f"Fit test function loss, AVG RMS: {avg_test_rms:.6f}, STD RMS: {std_test_rms:.6f}")
        ax[1][1].plot(x_test, avg_y_pred_test - y_test, 'b-o', label=f"Avg Test Loss")

        ax[2][0].set_title(f"Spectrum")
        ax[2][0].semilogy(np.abs(true_coef), 'r-o', label="True Coef")
        ax[2][0].semilogy(np.abs(avg_pred_coef), 'b-o', label=f"Avg Fit Coef")
        ax[2][0].set_ylim([1e-10, 10])
        # 可视化频谱的误差tuple
        ax[2][1].set_title(f"Spectrum Error, AVG RMS: {avg_spectrum_error:.6f}, STD RMS: {std_spectrum_error:.6f}")
        ax[2][1].semilogy(np.abs(avg_pred_coef - true_coef), 'b-o', label=f"Avg Fit Coef Error")

        # 收集所有seed的数据
        all_y_pred = []
        all_train_errors = []
        all_y_pred_test = []
        all_test_errors = []
        all_pred_coef = []
        all_coef_errors = []
        
        for seed in seeds:
            y_pred = results[seed][epoch]["y_pred"]
            pred_coef = results[seed][epoch]["pred_coef"]
            y_pred_test = results[seed][epoch]["y_pred_test"]
            
            all_y_pred.append(y_pred)
            all_train_errors.append(y_pred - y_train)
            all_y_pred_test.append(y_pred_test)
            all_test_errors.append(y_pred_test - y_test)
            all_pred_coef.append(np.abs(pred_coef))
            all_coef_errors.append(np.abs(pred_coef - true_coef))
        
        # 计算每个点的平均值和标准差
        mean_y_pred = np.mean(all_y_pred, axis=0)
        std_y_pred = np.std(all_y_pred, axis=0)
        
        mean_train_errors = np.mean(all_train_errors, axis=0)
        std_train_errors = np.std(all_train_errors, axis=0)
        
        mean_y_pred_test = np.mean(all_y_pred_test, axis=0)
        std_y_pred_test = np.std(all_y_pred_test, axis=0)
        
        mean_test_errors = np.mean(all_test_errors, axis=0)
        std_test_errors = np.std(all_test_errors, axis=0)
        
        mean_pred_coef = np.mean(all_pred_coef, axis=0)
        std_pred_coef = np.std(all_pred_coef, axis=0)
        
        mean_coef_errors = np.mean(all_coef_errors, axis=0)
        std_coef_errors = np.std(all_coef_errors, axis=0)
        
        # 使用fill_between绘制数据区间
        # 训练拟合结果
        ax[0][0].plot(x_train, mean_y_pred, '-', color='blue', label=f"Mean Fit sin(x)")
        ax[0][0].fill_between(x_train, 
                            mean_y_pred - std_y_pred, 
                            mean_y_pred + std_y_pred, 
                            color='blue', alpha=0.2)
        ax[0][0].legend()
        
        # 训练误差
        ax[0][1].plot(x_train, mean_train_errors, '-', color='green', label=f"Mean train Loss")
        ax[0][1].fill_between(x_train, 
                            mean_train_errors - std_train_errors, 
                            mean_train_errors + std_train_errors, 
                            color='green', alpha=0.2)
        ax[0][1].legend()
        
        # 测试拟合结果
        ax[1][0].plot(x_test, mean_y_pred_test, '-', color='red', label=f"Mean Fit sin(x)")
        ax[1][0].fill_between(x_test, 
                            mean_y_pred_test - std_y_pred_test, 
                            mean_y_pred_test + std_y_pred_test, 
                            color='red', alpha=0.2)
        ax[1][0].legend()
        
        # 测试误差
        ax[1][1].plot(x_test, mean_test_errors, '-', color='purple', label=f"Mean Test Loss")
        ax[1][1].fill_between(x_test, 
                            mean_test_errors - std_test_errors, 
                            mean_test_errors + std_test_errors, 
                            color='purple', alpha=0.2)
        ax[1][1].legend()
        
        # 频谱系数
        ax[2][0].semilogy(mean_pred_coef, '-', color='orange', label=f"Mean Fit Coef")
        ax[2][0].fill_between(range(len(mean_pred_coef)), 
                            mean_pred_coef - std_pred_coef, 
                            mean_pred_coef + std_pred_coef, 
                            color='orange', alpha=0.2)
        ax[2][0].legend()
        
        # 频谱误差
        ax[2][1].semilogy(mean_coef_errors, '-', color='brown', label=f"Mean Fit Coef Error")
        ax[2][1].fill_between(range(len(mean_coef_errors)), 
                            mean_coef_errors - std_coef_errors, 
                            mean_coef_errors + std_coef_errors, 
                            color='brown', alpha=0.2)
        ax[2][1].legend()


        fig.savefig(f"{output_dir}/beta_{beta}_epoch_{epoch}.png")
        plt.close(fig)  # 关闭当前figure，释放内存
        print(f"Training completed. Saved figures to {output_dir}/beta_{beta}_epoch_{epoch}.png")


def main():
    # 准备数据
    # 训练数据
    x_train = torch.linspace(DATA_RANGE[0], DATA_RANGE[1], 17)
    x_train = x_train.unsqueeze(1)
    y_train = TARGET_FUNC(x_train)
    # 测试数据
    x_test = torch.distributions.Uniform(DATA_RANGE[0], DATA_RANGE[1]).sample((100, 1))
    y_test = TARGET_FUNC(x_test)
    #对测试数据进行排序
    sorted_indices = torch.argsort(x_test[:, 0])
    x_test_sorted = x_test[sorted_indices]
    y_test_sorted = y_test[sorted_indices]

    # 验证分段线性插值
    # 把 tensor 转成 numpy
    x_train_np = x_train.numpy().flatten()
    y_train_np = y_train.numpy().flatten()
    x_test_np = x_test_sorted.numpy().flatten()
    y_test_np = y_test.numpy().flatten()

    # 构建插值函数
    f_linear = interp1d(x_train_np, y_train_np, kind='linear', fill_value="extrapolate")
    y_pred_linear = f_linear(x_test_np)

    # 计算线性插值在测试集上的 RMS 误差
    linear_test_rms = np.sqrt(np.mean((y_pred_linear - y_test_np)**2))
    print(f"🌟 Linear Interpolation Test RMS: {linear_test_rms:.6f}")

    # 移动数据到GPU
    x_train, y_train, x_test, y_test = x_train.to(DEVICE), y_train.to(DEVICE), x_test_sorted.to(DEVICE), y_test_sorted.to(DEVICE)
    
    # 初始化基线结果字典 - 符合微调脚本要求的结构
    results_base = {
        'x_train': x_train.cpu().numpy(),
        'y_train': y_train.cpu().numpy(),
        'x_test': x_test.cpu().numpy(),
        'y_test': y_test.cpu().numpy()
    }

    # 利用正交多项式计算频谱
    normalized_target = lambda x: TARGET_FUNC(torch.tensor(rescale(x, DATA_RANGE), dtype=torch.float32)).cpu().detach().numpy().flatten()
    true_coef = get_fq_coef(normalized_target)

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

        std_init = None
        
        for seed in SEEDS:
            print(f"Training beta={beta}, seed={seed}")
            set_seed(seed)
            results[seed] = {}
            rms_data[beta][seed] = {}
            baseline_metrics[beta][seed] = {}

            # 搭建模型
            model = FNNModel(n=100, beta=beta)
            std_init = model.layers[0].weight.std().item()
            model.to(DEVICE)

            # 训练模型
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            num_epochs = EPOCHS
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                y_pred = model(x_train)
                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()

                if (epoch+1) % 1000 == 0:
                    rms_data[beta][seed][epoch + 1] = {}
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

                    # 测试模型
                    model.eval()
                    with torch.no_grad():
                        y_pred_test = model(x_test)
                        test_loss = criterion(y_pred_test, y_test)
                    
                    # 计算模型预测值的正交多项式系数
                    normalized_pred = lambda x: model(torch.tensor(rescale(x, DATA_RANGE), dtype=torch.float32).reshape(-1, 1).to(DEVICE)).cpu().detach().numpy().flatten() # 确保输入是2D张量，并将输出转换为NumPy数组
                    pred_coef = get_fq_coef(normalized_pred)

                    # 将要可视化的数据存储到results中
                    results[seed][epoch + 1] = {}
                    results[seed][epoch + 1]["pred_coef"] = pred_coef
                    results[seed][epoch + 1]["y_pred"] = y_pred.cpu().detach().numpy().flatten()
                    results[seed][epoch + 1]["y_pred_test"] = y_pred_test.cpu().detach().numpy().flatten()

                    # 计算所有的误差
                    train_rms = np.sqrt(loss.item())
                    test_rms = np.sqrt(test_loss.item())
                    spectrum_error = np.sqrt(np.mean((pred_coef - true_coef)**2))

                    results[seed][epoch + 1]["train_rms"] = train_rms
                    results[seed][epoch + 1]["test_rms"] = test_rms
                    results[seed][epoch + 1]["spectrum_error"] = spectrum_error
                    rms_data[beta][seed][epoch + 1]["train_rms"] = train_rms
                    rms_data[beta][seed][epoch + 1]["test_rms"] = test_rms
                    rms_data[beta][seed][epoch + 1]["spectrum_error"] = spectrum_error
                    
                    # 保存基线指标 - 符合微调脚本要求的格式
                    baseline_metrics[beta][seed][epoch + 1] = {
                        "test_rms_base": test_rms,
                        "spectrum_error_base": spectrum_error,
                        "y_pred_test_base": y_pred_test.cpu().detach().numpy().flatten(),
                        "y_pred_train_base": y_pred.cpu().detach().numpy().flatten(),
                        "pred_coef_base": pred_coef
                    }
                    
                    # 保存模型 - 如果达到基线epoch
                    if epoch + 1 == BASELINE_EPOCH:
                        model_path = os.path.join(MODELS_DIR, f"model_beta_{beta}_seed_{seed}_epoch_{BASELINE_EPOCH}.pth")
                        torch.save(model.state_dict(), model_path)
                        print(f"Saved model to {model_path}")

        # 可视化训练结果
        plot_each_epoch(results, x_train.cpu().numpy().flatten(), y_train.cpu().numpy().flatten(), x_test.cpu().numpy().flatten(), y_test.cpu().numpy().flatten(), true_coef, beta, output_dir , std=std_init)
    

    # 利用rms_list绘制beta与rms的关系图
    fig_seed, ax_seed = plt.subplots(3, 1, figsize=(12, 18), sharex=True) # 共享x轴
    # 计算每个beta值下每个epoch的平均rms值和标准差
    avg_rms_data = {}
    std_rms_data = {}
    for beta in BETA:
        avg_rms_data[beta] = {}
        std_rms_data[beta] = {}
        for epoch in rms_data[beta][SEEDS[0]]:
            avg_rms_data[beta][epoch] = {}
            std_rms_data[beta][epoch] = {}
            # 计算平均值
            avg_rms_data[beta][epoch]["train_rms"] = np.mean([rms_data[beta][seed][epoch]["train_rms"] for seed in SEEDS])
            avg_rms_data[beta][epoch]["test_rms"] = np.mean([rms_data[beta][seed][epoch]["test_rms"] for seed in SEEDS])
            avg_rms_data[beta][epoch]["spectrum_error"] = np.mean([rms_data[beta][seed][epoch]["spectrum_error"] for seed in SEEDS])
            # 计算标准差
            std_rms_data[beta][epoch]["train_rms"] = np.std([rms_data[beta][seed][epoch]["train_rms"] for seed in SEEDS])
            std_rms_data[beta][epoch]["test_rms"] = np.std([rms_data[beta][seed][epoch]["test_rms"] for seed in SEEDS])
            std_rms_data[beta][epoch]["spectrum_error"] = np.std([rms_data[beta][seed][epoch]["spectrum_error"] for seed in SEEDS])

    for beta in BETA:
        epochs_recorded = sorted(avg_rms_data[beta].keys())
        # 获取平均值
        rms_value = np.array([avg_rms_data[beta][epoch]['train_rms'] for epoch in epochs_recorded])
        test_rms_value = np.array([avg_rms_data[beta][epoch]['test_rms'] for epoch in epochs_recorded])
        spectrum_error_value = np.array([avg_rms_data[beta][epoch]['spectrum_error'] for epoch in epochs_recorded])
        # 获取标准差
        rms_std = np.array([std_rms_data[beta][epoch]['train_rms'] for epoch in epochs_recorded])
        test_rms_std = np.array([std_rms_data[beta][epoch]['test_rms'] for epoch in epochs_recorded])
        spectrum_error_std = np.array([std_rms_data[beta][epoch]['spectrum_error'] for epoch in epochs_recorded])

        # 绘制平均线
        ax_seed[0].semilogy(epochs_recorded, rms_value,"-o", label=f"Beta={beta}", color=color_map[beta])
        ax_seed[1].semilogy(epochs_recorded, test_rms_value,"-o", label=f"Beta={beta}", color=color_map[beta])
        ax_seed[2].semilogy(epochs_recorded, spectrum_error_value,"-o", label=f"Beta={beta}", color=color_map[beta])
        
        # 使用fill_between绘制误差区间
        ax_seed[0].fill_between(epochs_recorded, rms_value - rms_std, rms_value + rms_std, 
                               color=color_map[beta], alpha=0.2)
        ax_seed[1].fill_between(epochs_recorded, test_rms_value - test_rms_std, test_rms_value + test_rms_std, 
                               color=color_map[beta], alpha=0.2)
        ax_seed[2].fill_between(epochs_recorded, spectrum_error_value - spectrum_error_std, spectrum_error_value + spectrum_error_std, 
                               color=color_map[beta], alpha=0.2)

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
    ax_seed[2].set_title(f"Spectrum Error RMS vs Epoch")
    ax_seed[2].set_xlabel("Epoch")
    ax_seed[2].set_ylabel("Spectrum Error RMS")
    ax_seed[2].legend()
    ax_seed[2].grid(which='both', linestyle='--', linewidth=0.5)

    fig_seed.savefig(f"{OUTPUT_DIR}/beta_rms_seed.png")
    plt.close(fig_seed)  # 关闭当前figure，释放内存
    print(f"RMS plot saved to {OUTPUT_DIR}/beta_rms_seed.png")
    
    # 保存基线结果 - 包含所有必要的信息供微调脚本使用
    results_base['BETA'] = BETA
    results_base['SEEDS'] = SEEDS
    results_base['BASELINE_EPOCH'] = BASELINE_EPOCH
    results_base['true_coef'] = true_coef
    results_base['metrics'] = baseline_metrics
    
    # 保存基线结果到pkl文件
    baseline_file = os.path.join(OUTPUT_DIR, "results_base.pkl")
    with open(baseline_file, 'wb') as f:
        pickle.dump(results_base, f)
    print(f"Saved baseline results to {baseline_file}")
    
    # 计算每个 beta 值在最后一个 epoch 的平均性能指标
    beta_values = []
    avg_test_rms = []
    std_test_rms = []
    avg_spectrum_error = []
    std_spectrum_error = []
    
    # 遍历每个 beta 值
    for beta in BETA:
        beta_values.append(beta)
        # 提取所有种子在最后一个 epoch 的 test_rms 和 spectrum_error
        final_test_rms_list = [rms_data[beta][seed][EPOCHS]['test_rms'] for seed in SEEDS]
        final_spectrum_error_list = [rms_data[beta][seed][EPOCHS]['spectrum_error'] for seed in SEEDS]
        
        # 计算平均值和标准差
        avg_test_rms.append(np.mean(final_test_rms_list))
        std_test_rms.append(np.std(final_test_rms_list))
        avg_spectrum_error.append(np.mean(final_spectrum_error_list))
        std_spectrum_error.append(np.std(final_spectrum_error_list))
    
    # 创建图形，绘制带填充区间的折线图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 子图1：Final Test RMS vs Beta
    ax1.plot(beta_values, avg_test_rms, 'o-', color='blue', label='Final Test RMS')
    ax1.fill_between(beta_values, 
                    avg_test_rms - np.array(std_test_rms), 
                    avg_test_rms + np.array(std_test_rms), 
                    color='blue', alpha=0.2)
    ax1.set_xlabel('Beta')
    ax1.set_ylabel('Final Test RMS')
    ax1.set_xticks(beta_values)
    ax1.set_yscale('log')
    ax1.set_title('Final Test RMS vs Beta')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 标记最低点对应的 Beta
    min_test_rms_idx = np.argmin(avg_test_rms)
    ax1.annotate(f'Best Beta: {beta_values[min_test_rms_idx]}', 
                xy=(beta_values[min_test_rms_idx], avg_test_rms[min_test_rms_idx]),
                xytext=(beta_values[min_test_rms_idx], avg_test_rms[min_test_rms_idx] * 1.5),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                fontsize=12, color='red')
    
    # 子图2：Final Spectrum Error vs Beta
    ax2.plot(beta_values, avg_spectrum_error, 'o-', color='green', label='Final Spectrum Error')
    ax2.fill_between(beta_values, 
                    avg_spectrum_error - np.array(std_spectrum_error), 
                    avg_spectrum_error + np.array(std_spectrum_error), 
                    color='green', alpha=0.2)
    ax2.set_xlabel('Beta')
    ax2.set_ylabel('Final Spectrum Error')
    ax2.set_xticks(beta_values)
    ax2.set_yscale('log')
    ax2.set_title('Final Spectrum Error vs Beta')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 标记最低点对应的 Beta
    min_spectrum_error_idx = np.argmin(avg_spectrum_error)
    ax2.annotate(f'Best Beta: {beta_values[min_spectrum_error_idx]}', 
                xy=(beta_values[min_spectrum_error_idx], avg_spectrum_error[min_spectrum_error_idx]),
                xytext=(beta_values[min_spectrum_error_idx], avg_spectrum_error[min_spectrum_error_idx] * 1.5),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
                fontsize=12, color='red')
    
    # 设置 Beta 轴为对数坐标
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    
    plt.tight_layout()
    
    # 保存图形
    output_path = os.path.join(OUTPUT_DIR, 'final_performance_vs_beta.png')
    plt.savefig(output_path)
    print(f"Saved final performance plot to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
