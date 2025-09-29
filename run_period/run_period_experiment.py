import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullLocator, MaxNLocator

# ==============================================================================
# 1. 实验配置
# ==============================================================================
# --- 设备配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 固定的网络参数 ---
FIXED_WIDTH = np.pi
FIXED_BETA = np.abs(np.log(np.sqrt(2)-1)) / FIXED_WIDTH
N_NEURONS_LIST = [32, 64, 128, 512]
LEARNING_RATE = 0.01
EPOCHS_LIST = [100,1000]
epochs = EPOCHS_LIST[-1]
SEED_LIST = [42, 38, 50, 100]

# --- 扫描的周期参数 ---# 核心修改：以period为主参数
PERIOD_LIST = np.array([0.5, 1.5, 2.0, 2.5, 3.0, 4.0]) * FIXED_WIDTH
K_LIST = (2 * np.pi) / PERIOD_LIST


# --- 数据集参数 ---
DATA_RANGE = [-2 * np.pi, 2 * np.pi]
NUM_POINTS = 1000
TEST_SIZE = 0.2

# --- 输出配置 ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "experiment_results_period_Π_wide")

# ==============================================================================
# 2. 模型定义
# ==============================================================================
# 简单的全连接神经网络
class SimpleMLP(nn.Module):
    def __init__(self, n_neurons, beta):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, n_neurons),
            nn.Softplus(beta), # default:torch.nn.Softplus(beta=1.0, threshold=20)
            nn.Linear(n_neurons, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# ==============================================================================
# 3. 修改数据生成函数
# ==============================================================================
def generate_data_by_period(period, data_range, num_points, test_size):
    """
    生成并分割不同周期的sin波数据集
    Args:
        period (float): 周期参数
        data_range (list): 数据范围，例如 [-2*pi, 2*pi]
        num_points (int): 数据点数量
        test_size (float): 测试集比例
    Returns:
        X_train_t, y_train_t, X_test_t, y_test_t: 训练集和测试集的 PyTorch Tensors
    """
    k = 2 * np.pi / period  # 从周期计算频率
    X = np.linspace(data_range[0], data_range[1], num_points).reshape(-1, 1)
    y = np.sin(k * X)  # 使用计算出的频率生成sin波
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # 转换为PyTorch Tensors并移动到设备
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_test_t = torch.FloatTensor(y_test).to(DEVICE)
    
    return X_train_t, y_train_t, X_test_t, y_test_t

# ==============================================================================
# 4. 训练与评估函数
# ==============================================================================
def train_and_evaluate(model, X_train, y_train, X_test, y_test, lr, epochs, n, seed, output_dir, current_period):
    """
    训练模型，评估并绘制损失曲线
    Args:
        model (nn.Module): 模型实例
        X_train (torch.Tensor): 训练集输入
        y_train (torch.Tensor): 训练集标签
        X_test (torch.Tensor): 测试集输入
        y_test (torch.Tensor): 测试集标签
        lr (float): 学习率
        epochs (int): 训练轮数
        n (int): 模型神经元数量
        seed (int): 随机种子
        output_dir (str): 图像输出目录
        current_period (float): 当前的周期参数
    Returns:
        std_dev (float): 测试集上的误差标准差
        y_pred (torch.Tensor): 模型在测试集上的预测值
    """
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train) # shape: (batch_size, 1)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # 每隔10轮记录一次损失
        if epoch % 10 == 0:
            train_losses.append(loss.item())
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test)
                val_losses.append(val_loss.item())

            print(f"Epoch [{epoch}/{epochs}], n={n}, period={current_period/np.pi:.2f}π, seed={seed} Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
        
        # 如果当前轮次是epochlist中的一个元素
        if epoch + 1 in EPOCHS_LIST:
            # 计算当前轮次的预测值
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                error = y_test - y_pred
                std_dev = error.std().item()
                results['period_results'][epoch + 1][current_period][n][seed]['y_pred'] = y_pred.cpu().numpy().flatten()
                results['period_results'][epoch + 1][current_period][n][seed]['y_pred_std'] = std_dev

    # --- 绘制并保存当前参数下的损失曲线 ---
    epochs_recorded = np.arange(len(train_losses)) * 10 
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.plot(epochs_recorded, train_losses, label='Train Loss')
    ax.plot(epochs_recorded, val_losses, label='Validation Loss')
    # 修改标题，使用period而不是k
    ax.set_title(f'Loss Curve for n={n}, period={current_period/np.pi:.2f}π')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    ax.set_yscale('log')
    # 合并训练损失和验证损失，去除重复值并排序，以此设置 y 轴刻度
    all_losses = sorted(set(train_losses + val_losses))
    ax.set_yticks(all_losses)
    # 对y轴的刻度倾斜45度
    ax.tick_params(axis='y', labelrotation=45)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.legend()
    ax.grid(True)
    
    # 创建特定period值的目录
    period_dir = os.path.join(output_dir, f'period_{current_period / np.pi}Π')
    if not os.path.exists(period_dir):
        os.makedirs(period_dir)
    
    loss_curve_path = os.path.join(period_dir, f'loss_curve_n{n}_seed{seed}.png')
    ax.figure.savefig(loss_curve_path)
    plt.close(ax.figure)
    
    # --- 在测试集上计算最终误差的标准差 ---
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        error = y_test - y_pred
        std_dev = error.std().item()
        
    return std_dev, y_pred

# ==============================================================================
# 5. 主实验流程
# ==============================================================================
if __name__ == "__main__":
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 初始化results字典
    results = {
        'fixed_beta': FIXED_BETA,
        'fixed_width': FIXED_WIDTH,
        'k_list': K_LIST,
        'period_list': PERIOD_LIST,
        'period_results': {
            epoch: {
                # 结果将按 period 存储
                period: {
                    n: {
                        seed: {'y_pred_std': None, 'y_pred': None} for seed in SEED_LIST
                    } for n in N_NEURONS_LIST
                } for period in PERIOD_LIST
            } for epoch in EPOCHS_LIST
        }
    }

    # 核心修改：外层循环直接遍历 period 值，而不是k值
    for current_period in PERIOD_LIST:
        # 从周期计算频率（如果需要）
        k = 2 * np.pi / current_period
        print(f"\n{'='*20} Running for period={current_period/np.pi:.2f}π (k={k:.4f}) {'='*20}")

        # 1. 使用基于period的数据生成函数
        X_train, y_train, X_test, y_test = generate_data_by_period(current_period, DATA_RANGE, NUM_POINTS, TEST_SIZE)
        
        # 存储当前period值的测试数据，使用period作为键的一部分
        results[f'X_test_period_{current_period}'] = X_test.cpu().numpy()
        results[f'y_test_period_{current_period}'] = y_test.cpu().numpy()
        results[f'X_train_period_{current_period}'] = X_train.cpu().numpy()
        results[f'y_train_period_{current_period}'] = y_train.cpu().numpy()

        # 内层循环遍历 n 和 seed
        for n in N_NEURONS_LIST:
            print(f"\n--- Training with n={n} --- ")
            
            for seed in SEED_LIST:
                # 设置随机种子
                print(f"Setting seed to {seed} for n={n}, period={current_period/np.pi:.2f}π")
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                # 2. 模型现在总是使用固定的 beta
                model = SimpleMLP(n_neurons=n, beta=FIXED_BETA).to(DEVICE)
                
                # 3. 训练和评估
                std_dev, y_pred = train_and_evaluate(
                    model, X_train, y_train, X_test, y_test, LEARNING_RATE, epochs, n, seed, OUTPUT_DIR, current_period
                )
                
                
                print(f"--- Result for n={n}, period={current_period/np.pi:.2f}π, seed={seed}: std(y_err) = {std_dev:.6f} ---")
    
    # 将results保存为pickle文件
    with open(os.path.join(OUTPUT_DIR, f'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*50}")
    print(f"实验完成！结果已保存到 {OUTPUT_DIR}/results.pkl")
    print(f"{'='*50}")