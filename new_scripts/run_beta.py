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

BETA = [1.0, 4.0, 8.0, 16.0]
TARGET_FUNC = lambda x: torch.sin(x)
# TARGET_FUNC = lambda x: (x+0.8) * np.arcsin(np.sin(2*x*x+5*x))
DATA_RANGE = [-2 * np.pi, 2 * np.pi]
EPOCHS = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)


OUTPUT_DIR = f"figures/beta/"
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

def main():
    # 准备数据
    # 训练数据
    x_train = torch.linspace(DATA_RANGE[0], DATA_RANGE[1], 17)
    x_train = x_train.unsqueeze(1)
    y_train = TARGET_FUNC(x_train)
    # 测试数据
    x_test = torch.distributions.Uniform(DATA_RANGE[0], DATA_RANGE[1]).sample((100, 1))
    y_test = TARGET_FUNC(x_test)
    # 移动数据到GPU
    x_train, y_train, x_test, y_test = x_train.to(DEVICE), y_train.to(DEVICE), x_test.to(DEVICE), y_test.to(DEVICE)

    # 利用正交多项式计算频谱
    normalized_target = lambda x: TARGET_FUNC(torch.tensor(rescale(x, DATA_RANGE), dtype=torch.float32)).cpu().detach().numpy().flatten()
    true_coef = get_fq_coef(normalized_target)


    #将每个beta下每个模型的训练结果rms存储在一个列表中
    rms_data = {}
    # 为每一个beta值训练一个模型
    for beta in BETA:
        rms_data[beta] = {}
        #定义输出目录
        output_dir = f"{OUTPUT_DIR}/beta_{beta}/"
        os.makedirs(output_dir, exist_ok=True)

        # 搭建模型
        model = FNNModel(n=100, beta=beta)
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
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
                # 可视化训练结果
                fig, ax = plt.subplots(3,2, figsize=(20, 15))
                # 测试模型
                model.eval()
                with torch.no_grad():
                    y_pred_test = model(x_test)
                    test_loss = criterion(y_pred_test, y_test)
                
                # 计算模型预测值的正交多项式系数
                normalized_pred = lambda x: model(torch.tensor(rescale(x, DATA_RANGE), dtype=torch.float32).reshape(-1, 1).to(DEVICE)).cpu().detach().numpy().flatten() # 确保输入是2D张量，并将输出转换为NumPy数组
                pred_coef = get_fq_coef(normalized_pred)

                # 可视化训练结果
                rms_data[beta][epoch + 1] = {}
                fig.suptitle(f"Softplus Activation with Beta={beta} Epoch {epoch+1}")
                ax[0][0].set_title(f"Fit train function with {x_train.shape[0]} points")
                ax[0][0].plot(x_train.cpu().numpy(), y_train.cpu().numpy(), 'r:o', label=f"True sin(x)")
                ax[0][0].plot(x_train.cpu().numpy(), y_pred.cpu().detach().numpy(), 'b-o', label=f"Fit sin(x)")
                ax[0][0].legend()
                train_rms = np.sqrt(loss.item())
                rms_data[beta][epoch + 1]["train_rms"] = train_rms
                ax[0][1].set_title(f"Fit train function loss, RMS: {train_rms:.6f}")
                ax[0][1].plot(x_train.cpu().numpy(), y_pred.cpu().detach().numpy() - y_train.cpu().numpy(), 'b-o', label=f"Train Loss")
                ax[0][1].legend()
                # 可视化测试结果
                # 对测试数据重新排序
                sorted_indices = torch.argsort(x_test[:, 0])
                x_test_sorted = x_test[sorted_indices]
                y_test_sorted = y_test[sorted_indices]
                y_pred_test_sorted = y_pred_test[sorted_indices]
                ax[1][0].set_title(f"Fit test function with {x_test.shape[0]} points")
                ax[1][0].plot(x_train.cpu().numpy(), y_train.cpu().numpy(), 'k-', label="Train sin(x)")
                ax[1][0].plot(x_test_sorted.cpu().numpy(), y_test_sorted.cpu().numpy(), 'r:o', label=f"True sin(x)")
                ax[1][0].plot(x_test_sorted.cpu().numpy(), y_pred_test_sorted.cpu().detach().numpy(), 'b-o', label=f"Fit sin(x)")
                ax[1][0].legend()
                test_rms = np.sqrt(test_loss.item())
                rms_data[beta][epoch + 1]["test_rms"] = test_rms
                ax[1][1].set_title(f"Fit test function loss, RMS: {test_rms:.6f}")
                ax[1][1].plot(x_test_sorted.cpu().numpy(), y_pred_test_sorted.cpu().detach().numpy() - y_test_sorted.cpu().numpy(), 'b-o', label=f"Test Loss")
                ax[1][1].legend()
                # 可视化频谱
                ax[2][0].set_title(f"Spectrum")
                ax[2][0].semilogy(np.abs(true_coef), 'r-o', label="True Coef")
                ax[2][0].semilogy(np.abs(pred_coef), 'b-o', label=f"Fit Coef")
                ax[2][0].legend()
                ax[2][0].set_ylim([1e-10, 10])
                # 可视化频谱的误差tuple
                spectrum_error = np.sqrt(np.mean((pred_coef - true_coef)**2))
                rms_data[beta][epoch + 1]["spectrum_error"] = spectrum_error
                ax[2][1].set_title(f"Spectrum Error, RMS: {spectrum_error:.6f}")
                ax[2][1].semilogy(np.abs(pred_coef - true_coef), 'b-o', label=f"Fit Coef Error")
                ax[2][1].legend()
                ax[2][1].set_ylim([1e-10, 10])

                fig.savefig(f"{output_dir}/beta_{beta}_epoch_{epoch+1}.png")
                plt.close(fig)  # 关闭当前figure，释放内存
        print(f"Training completed. Saved figures to {output_dir}")

    # 利用rms_list绘制beta与rms的关系图
    fig, ax = plt.subplots(3, 1, figsize=(12, 18), sharex=True) # 共享x轴
    for beta in BETA:
        epochs_recorded = sorted(rms_data[beta].keys())
        rms_value = np.array(list(rms_data[beta][epoch]['train_rms']for epoch in epochs_recorded))
        test_rms_value = np.array(list(rms_data[beta][epoch]['test_rms']for epoch in epochs_recorded))
        spectrum_error_value = np.array(list(rms_data[beta][epoch]['spectrum_error']for epoch in epochs_recorded))

        ax[0].semilogy(epochs_recorded, rms_value, label=f"Beta={beta}")
        ax[1].semilogy(epochs_recorded, test_rms_value, label=f"Beta={beta}")
        ax[2].semilogy(epochs_recorded, spectrum_error_value, label=f"Beta={beta}")
    ax[0].set_title(f"Train RMS vs Epoch")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Train RMS")
    ax[0].legend()
    ax[0].grid(which='both', linestyle='--', linewidth=0.5)
    ax[1].set_title(f"Test RMS vs Epoch")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Test RMS")
    ax[1].legend()
    ax[1].grid(which='both', linestyle='--', linewidth=0.5)
    ax[2].set_title(f"Spectrum Error RMS vs Epoch")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Spectrum Error RMS")
    ax[2].legend()
    ax[2].grid(which='both', linestyle='--', linewidth=0.5)
    fig.savefig(f"{OUTPUT_DIR}/beta_rms.png")
    plt.close(fig)  # 关闭当前figure，释放内存
    print(f"RMS plot saved to {OUTPUT_DIR}/beta_rms.png")


if __name__ == "__main__":
    main()
