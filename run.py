import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from matplotlib.ticker import ScalarFormatter 
import pickle

# ==============================================================================
# 1. 全局配置参数 (All parameters are here for easy adjustment)
# ==============================================================================

# --- 设备配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 数据集参数 ---
DATA_RANGE = [-2 * np.pi, 2 * np.pi]
NUM_POINTS = 1000
TEST_SIZE = 0.2

# --- 模型与训练参数 ---
# n也作为一个变量进行测试
N_NEURONS_LIST = [32]  # 您可以修改或增加神经元数量的测试列表
LEARNING_RATE = 0.01
EPOCHS_LIST = [100, 1000, 10000] # 实验不同训练时长的影响
epochs = EPOCHS_LIST[-1]

# --- 实验核心：β 扫描列表 ---
# 定义当前要使用的 BETA 列表
BETA_VISUALIZE = [0.5,10,500] # 指定需要可视化拟合曲线的beta值
BETA_BASE = [0.5,10,500]
BETA_TO_RUN = sorted(list(set(BETA_BASE + BETA_VISUALIZE))) # 合并并排序

# --- 输出配置 ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "experiment_results")
# # 设置随机种子以保证结果可复现
# torch.manual_seed(52)
# np.random.seed(52)

# --- 统计稳定性测试参数 ---
SEED_LIST = [42,38,100]  # 使用多个随机种子进行重复实验

# ==============================================================================
# 2. 模型定义 (Model Definition)
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
# 3. 数据准备函数 (Data Preparation Function)
# ==============================================================================

def generate_data(data_range, num_points, test_size):
    """
    生成并分割 sin(x) 数据集
    Args:
        data_range (list): 数据范围，例如 [-2*pi, 2*pi]
        num_points (int): 数据点数量
        test_size (float): 测试集比例
    Returns:
        X_train_t, y_train_t, X_test_t, y_test_t: 训练集和测试集的 PyTorch Tensors
    """
    X = np.linspace(data_range[0], data_range[1], num_points).reshape(-1, 1)
    y = np.sin(X)
    
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
# 4. 训练与评估函数 (Training and Evaluation Function)
# ==============================================================================

def train_and_evaluate(model, X_train, y_train, X_test, y_test, lr, epochs, n, beta, output_dir):
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
        beta (float): Softplus参数
        output_dir (str): 图像输出目录
    Returns:
        std_dev (float): 测试集上的误差标准差
        y_pred (torch.Tensor): 模型在测试集上的预测值
    """
    criterion = lambda y_pred, y_true: torch.std(y_true - y_pred)  # 标准差损失函数
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
        
        # 每隔10轮记录一次损失和准确率
        if epoch % 10 == 0:
            train_losses.append(loss.item())
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test)
                val_losses.append(val_loss.item())

            print(f"Epoch [{epoch}/{epochs}], n={n}, beta={beta}, seed={seed} Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

        # 如果当前轮次是epochlist中的一个元素
        if epoch + 1 in EPOCHS_LIST and beta in BETA_VISUALIZE:
            # 计算当前轮次的预测值
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test)
                error = y_test - y_pred
                std_dev = error.std().item()
                results['train_results'][epoch + 1][n][beta][seed]['y_pred'] = y_pred.cpu().numpy().flatten()
                results['train_results'][epoch + 1][n][beta][seed]['y_pred_std'] = std_dev

    # --- 绘制并保存当前参数下的损失曲线 ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Loss Curve for n={n}, beta={beta}')
    plt.xlabel('Epochs (x10)')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.grid(True)
    loss_curve_path = os.path.join(output_dir, f'loss_curve_n{n}_beta{beta}.png')
    plt.savefig(loss_curve_path)
    plt.close()
    
    # --- 在测试集上计算最终误差的标准差 ---
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        error = y_test - y_pred
        std_dev = error.std().item()
        
    return std_dev, y_pred


# ==============================================================================
# 5. 主实验流程 (Main Experiment Logic)
# ==============================================================================

if __name__ == "__main__":
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 生成数据
    X_train, y_train, X_test, y_test = generate_data(DATA_RANGE, NUM_POINTS, TEST_SIZE)
    
    # 调整results结构以容纳epochs, mean_std, 和 std_of_stds
    results = {
        # 直接将测试数据存进去 (注意从Tensor转为Numpy array)
        'X_test': X_test.cpu().numpy(),
        'y_test': y_test.cpu().numpy(),
        
        # 将原来的训练结果嵌套在一个新的键 'train_results' 中
        'train_results': {
            epochs: {
                n: {
                    beta: {
                        seed: {'y_pred_std': None, 'y_pred': None} for seed in SEED_LIST
                    } for beta in BETA_VISUALIZE
                } for n in N_NEURONS_LIST
            } for epochs in EPOCHS_LIST
        }
    }
    for n in N_NEURONS_LIST:
        print(f"\n{'='*20} Starting Experiment for n = {n} Neurons {'='*20}")
        
        for beta in BETA_TO_RUN:
            print(f"\n--- Training with n={n}, beta={beta} ---")

            for seed in SEED_LIST:
                # 设置随机种子
                print(f"Setting seed to {seed} for n={n}, beta={beta}")
                torch.manual_seed(seed)
                np.random.seed(seed)
            
                # 实例化模型
                model = SimpleMLP(n_neurons=n, beta=beta).to(DEVICE)
                
                # 训练和评估
                std_dev, y_pred = train_and_evaluate(
                    model, X_train, y_train, X_test, y_test, LEARNING_RATE, epochs, n, beta, OUTPUT_DIR
                )

                print(f"--- Result for n={n}, beta={beta}, seed={seed}: std(y_err) = {std_dev:.6f} ---")


    # 将results保存为pickle文件
    with open(os.path.join(OUTPUT_DIR, f'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
