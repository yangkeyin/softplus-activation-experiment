import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from matplotlib.ticker import ScalarFormatter 
import pickle

from numpy.polynomial import legendre as lege

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.weighted_orthpoly_solver import orthpoly_coef

# ==============================================================================
# 1. 全局配置参数 (All parameters are here for easy adjustment)
# ==============================================================================

# --- 设备配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 数据集参数 ---
DATA_RANGE = [-6 * np.pi, 6 * np.pi]
NUM_POINTS = 1000
TEST_SIZE = 0.2

# --- 模型与训练参数 ---
# n也作为一个变量进行测试
N_NEURONS = 64  # 您可以修改或增加神经元数量的测试列表
LEARNING_RATE = 0.001

# --- 核心修改部分 ---
# 定义当前要使用的 BETA 列表
BETA_TO_RUN = [0.5,1,2,4,8,10,20,50] # 指定需要可视化拟合曲线的beta值
EPOCHS_LIST = [100, 1000] # 实验不同训练时长的影响, 取最后一个值进行实验

# --- 输出配置 ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../results/demo")
# # 设置随机种子以保证结果可复现
# torch.manual_seed(52)
# np.random.seed(52)

# --- 统计稳定性测试参数 ---
SEED_LIST = [42,38,100]  # 使用多个随机种子进行重复实验

# --- 勒让德多项式相关参数 ---
N_POLY_ORDER = 100 # 勒让德多项式的阶数
V = orthpoly_coef(None, 'legendre', N_POLY_ORDER)
#x_sample = sin( pi * (-n_poly_order/2 + arange(n_poly_order+1))/(n_poly_order+1) )
x_sample = lege.legroots(1*(np.arange(N_POLY_ORDER+2)==N_POLY_ORDER+1)) # 生成n_poly_order+2个项的勒让德多项式多项式，最高项系数为1，并求解其根
get_fq_coef = lambda f: np.linalg.solve(V, f(x_sample)) # 函数，解线性方程组V * c = f(x_sample)的系数，这组系数就是函数f的频谱，c[0]代表最平缓的成分，c[k]代表越来越高频，变化越来越剧烈的部分

# --- 频谱线性变换函数 ---
a, b = DATA_RANGE[0], DATA_RANGE[1]
rescale_func = lambda x_norm: (b - a) / 2. * x_norm + (a + b) / 2.


# ==============================================================================
# 2. 模型定义 (Model Definition)
# ==============================================================================
# 简单的全连接神经网络
class SimpleMLP(nn.Module):
    def __init__(self, beta):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.Softplus(beta), # default:torch.nn.Softplus(beta=1.0, threshold=20)
            nn.Linear(64, 1)
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

def train_and_evaluate(model, X_train, y_train, X_test, y_test, lr, epochs, beta, output_dir, seed, results):
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
        seed (int): 当前随机种子
        results (dict): 结果字典，用于存储中间结果
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
        
        # 每隔10轮记录一次损失和准确率
        if epoch % 10 == 0:
            train_losses.append(loss.item())
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test)
                val_losses.append(val_loss.item())

            print(f"Epoch [{epoch}/{epochs}], beta={beta}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

        # 如果当前轮次是epochlist中的一个元素
        if epoch + 1 in EPOCHS_LIST:
            # 计算当前轮次的预测值
            model.eval()
            with torch.no_grad():
                # 记录预测值和误差标准差
                y_pred = model(X_test)
                error = y_test - y_pred
                std_dev = error.std().item()
                results['train_results'][epoch + 1][beta][seed]['y_pred'] = y_pred.cpu().numpy().flatten()
                results['train_results'][epoch + 1][beta][seed]['y_pred_std'] = std_dev

                # 记录当前轮次的频域
                f_ann = lambda x_norm: model(torch.FloatTensor(rescale_func(x_norm)).reshape(-1, 1).to(DEVICE)).cpu().numpy().flatten()
                ann_coeffs = get_fq_coef(f_ann)
                results['train_results'][epoch + 1][beta][seed]['ann_coeffs'] = ann_coeffs

    # --- 绘制并保存当前参数下的损失曲线 ---
    epochs_recorded = np.arange(len(train_losses)) * 10 
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.plot(epochs_recorded, train_losses, label='Train Loss')
    ax.plot(epochs_recorded, val_losses, label='Validation Loss')
    ax.set_title(f'Loss Curve for beta={beta}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    ax.set_yscale('log')
    
    # 修复网格线问题：分开设置主刻度和副刻度网格线样式
    ax.grid(True, which='major', linestyle='-', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.5)
    
    # 修复副刻度显示数字的问题
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_minor_formatter(FormatStrFormatter('%.6g'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.6g'))
    ax.tick_params(axis='y', which='minor', labelsize=8, labelcolor='gray')

    ax.legend()
    loss_curve_path = os.path.join(output_dir, f'loss_curve_beta{beta}_seed{seed}.png')
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
# 5. 主实验流程 (Main Experiment Logic)
# ==============================================================================

if __name__ == "__main__":
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 生成数据
    X_train, y_train, X_test, y_test = generate_data(DATA_RANGE, NUM_POINTS, TEST_SIZE)

    # 获得真实函数的频域系数
    true_function_rescaled = lambda x_norm: np.sin(rescale_func(x_norm))
    # 在[-1, 1]上对包装后的函数进行频谱分析
    true_coeffs = get_fq_coef(true_function_rescaled)
    
    # 调整results结构以容纳epochs, mean_std, 和 std_of_stds
    results = {
        # 直接将测试数据存进去 (注意从Tensor转为Numpy array)
        'X_test': X_test.cpu().numpy(),
        'y_test': y_test.cpu().numpy(),
        'X_train': X_train.cpu().numpy(),
        'y_train': y_train.cpu().numpy(),
        'true_coeffs': true_coeffs,
        'fixed_params': {
            'beta': BETA_TO_RUN,
            'epochs': EPOCHS_LIST,
            'seed': SEED_LIST,
        },

        # 将原来的训练结果嵌套在一个新的键 'train_results' 中
        'train_results': {
            epochs: {
                    beta: {
                        seed: {'y_pred_std': None, 'y_pred': None, 'ann_coeffs': None} for seed in SEED_LIST
                    } for beta in BETA_TO_RUN
            } for epochs in EPOCHS_LIST
        }
    }
    
    for beta in BETA_TO_RUN:
        print(f"\n--- Training with beta={beta} ---")

        for seed in SEED_LIST:
            # 设置随机种子
            print(f"Setting seed to {seed} for beta={beta}")
            torch.manual_seed(seed)
            np.random.seed(seed)
        
            # 实例化模型
            model = SimpleMLP(beta=beta).to(DEVICE)
            
            # 训练和评估
            std_dev, y_pred = train_and_evaluate(
                model, X_train, y_train, X_test, y_test, LEARNING_RATE, EPOCHS_LIST[-1], beta, OUTPUT_DIR, seed, results
            )

        
        print(f"--- Result for beta={beta}: std(y_err) = {std_dev:.6f} ---")


    # 将results保存为pickle文件
    with open(os.path.join(OUTPUT_DIR, f'results.pkl'), 'wb') as f:
        pickle.dump(results, f)