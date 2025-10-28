from re import X
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.validation import sp
from sympy import li
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import rescale, get_fq_coef

BETA = [1.0, 2.0, 4.0, 6.0, 8.0, 16.0]
TARGET_FUNC = lambda x: torch.sin(x)
# TARGET_FUNC = lambda x: (x+0.8) * np.arcsin(np.sin(2*x*x+5*x))
DATA_RANGE = [-2 * np.pi, 2 * np.pi]
EPOCHS = 10000
SEEDS = [100, 200, 300]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


OUTPUT_DIR = f"figures/optimizer/Adagrad_lr0.1_1028/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class FNNModel(nn.Module):
    def __init__(self, n, beta):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, n),
            nn.Softplus(beta=beta),
            nn.Linear(n, 1)
        )
    def forward(self, x):
        return self.layers(x)
        
def set_seed(seed_value):
    """设置所有需要随机种子的库的种子。"""
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    # Python的内置随机库
    import random
    random.seed(seed_value)


def plot_each_epoch(results, x_train, y_train, x_test, y_test, true_coef, beta, output_dir):
    seeds = list(results.keys())
    epochs = list(results[seeds[0]].keys())
    for epoch in epochs:
        # 可视化训练结果
        fig, ax = plt.subplots(3,2, figsize=(20, 15))
        fig.suptitle(f"Softplus Activation with Beta={beta} Epoch {epoch+1}")
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

        for seed in seeds:
            y_pred = results[seed][epoch]["y_pred"]
            pred_coef = results[seed][epoch]["pred_coef"]
            y_pred_test = results[seed][epoch]["y_pred_test"]

            ax[0][0].plot(x_train, y_pred, ':', label=f"Fit sin(x), seed={seed}",alpha=0.7)
            ax[0][0].legend()
            ax[0][1].plot(x_train, y_pred - y_train, ':', label=f"Avg train Loss, seed={seed}",alpha=0.7)
            ax[0][1].legend()
            # 可视化测试结果
            ax[1][0].plot(x_test, y_pred_test, ':', label=f"Fit sin(x), seed={seed}",alpha=0.7)   
            ax[1][0].legend()
            ax[1][1].plot(x_test, y_pred_test - y_test, ':', label=f"Test Loss, seed={seed}",alpha=0.7)
            ax[1][1].legend()
            # 可视化频谱
            ax[2][0].semilogy(np.abs(pred_coef), ':', label=f"Fit Coef, seed={seed}",alpha=0.7)
            ax[2][0].legend()
            # 可视化频谱的误差tuple
            ax[2][1].semilogy(np.abs(pred_coef - true_coef), ':', label=f"Fit Coef Error, seed={seed}",alpha=0.7)
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
    # 移动数据到GPU
    x_train, y_train, x_test, y_test = x_train.to(DEVICE), y_train.to(DEVICE), x_test_sorted.to(DEVICE), y_test_sorted.to(DEVICE)

    # 利用正交多项式计算频谱
    normalized_target = lambda x: TARGET_FUNC(torch.tensor(rescale(x, DATA_RANGE), dtype=torch.float32)).cpu().detach().numpy().flatten()
    true_coef = get_fq_coef(normalized_target)

    # 为每个beta都定义一个颜色
    beta_colors = plt.cm.tab10(np.linspace(0, 1, len(BETA)))
    color_map = {beta: beta_colors[i] for i, beta in enumerate(BETA)}


    #将每个beta下每个模型的训练结果rms存储在一个列表中
    rms_data = {}
    # 为每一个beta值训练一个模型
    for beta in BETA:
        print(f"Training beta={beta}")
        results = {}
        rms_data[beta] = {}
        #定义输出目录
        output_dir = f"{OUTPUT_DIR}/beta_{beta}/"
        os.makedirs(output_dir, exist_ok=True)
        for seed in SEEDS:
            print(f"Training beta={beta}, seed={seed}")
            set_seed(seed)
            results[seed] = {}
            rms_data[beta][seed] = {}

            # 搭建模型
            model = FNNModel(n=100, beta=beta)
            model.to(DEVICE)

            # 训练模型
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)
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

        # 可视化训练结果
        plot_each_epoch(results, x_train.cpu().numpy().flatten(), y_train.cpu().numpy().flatten(), x_test.cpu().numpy().flatten(), y_test.cpu().numpy().flatten(), true_coef, beta, output_dir)
    

    # 利用rms_list绘制beta与rms的关系图
    fig_seed, ax_seed = plt.subplots(3, 1, figsize=(12, 18), sharex=True) # 共享x轴
    # 计算每个beta值下每个epoch的平均rms值
    avg_rms_data = {}
    # 计算每个epoch下的train_rms、test_rms、spectrum_error
    epoch_rms_data = {}
    for beta in BETA:
        avg_rms_data[beta] = {}
        epoch_rms_data[beta] = {}
        for epoch in rms_data[beta][SEEDS[0]]:
            avg_rms_data[beta][epoch] = {}
            avg_rms_data[beta][epoch]["train_rms"] = np.mean([rms_data[beta][seed][epoch]["train_rms"] for seed in SEEDS])
            avg_rms_data[beta][epoch]["test_rms"] = np.mean([rms_data[beta][seed][epoch]["test_rms"] for seed in SEEDS])
            avg_rms_data[beta][epoch]["spectrum_error"] = np.mean([rms_data[beta][seed][epoch]["spectrum_error"] for seed in SEEDS])
        for seed in SEEDS:
            epoch_rms_data[beta][seed] = {}
            epoch_rms_data[beta][seed]["train_rms"] = np.array([rms_data[beta][seed][epoch]["train_rms"] for epoch in avg_rms_data[beta]])
            epoch_rms_data[beta][seed]["test_rms"] = np.array([rms_data[beta][seed][epoch]["test_rms"] for epoch in avg_rms_data[beta]])
            epoch_rms_data[beta][seed]["spectrum_error"] = np.array([rms_data[beta][seed][epoch]["spectrum_error"] for epoch in avg_rms_data[beta]])

    for beta in BETA:
        epochs_recorded = sorted(avg_rms_data[beta].keys())
        rms_value = np.array(list(avg_rms_data[beta][epoch]['train_rms']for epoch in epochs_recorded))
        test_rms_value = np.array(list(avg_rms_data[beta][epoch]['test_rms']for epoch in epochs_recorded))
        spectrum_error_value = np.array(list(avg_rms_data[beta][epoch]['spectrum_error']for epoch in epochs_recorded))

        ax_seed[0].semilogy(epochs_recorded, rms_value,"-o", label=f"Beta={beta}", color=color_map[beta])
        ax_seed[1].semilogy(epochs_recorded, test_rms_value,"-o", label=f"Beta={beta}", color=color_map[beta])
        ax_seed[2].semilogy(epochs_recorded, spectrum_error_value,"-o", label=f"Beta={beta}", color=color_map[beta])
        # for seed in SEEDS:
        #     ax_seed[0].semilogy(epochs_recorded, epoch_rms_data[beta][seed]["train_rms"], ':', color=color_map[beta], alpha=0.7)
        #     ax_seed[1].semilogy(epochs_recorded, epoch_rms_data[beta][seed]["test_rms"], ':', color=color_map[beta], alpha=0.7)
        #     ax_seed[2].semilogy(epochs_recorded, epoch_rms_data[beta][seed]["spectrum_error"], ':', color=color_map[beta], alpha=0.7)

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


if __name__ == "__main__":
    main()
