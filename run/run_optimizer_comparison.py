import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
import pickle
from matplotlib.ticker import FormatStrFormatter, NullLocator, MaxNLocator
from numpy.polynomial import legendre as lege

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.weighted_orthpoly_solver import orthpoly_coef



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
# 固定参数以隔离优化器的影响
N_NEURONS = 128  # 固定为128神经元
BETA = 2.0  # 固定为2
EPOCHS_LIST = [100, 1000,2000,3000,4000,5000] # 实验不同训练时长的影响
epochs = EPOCHS_LIST[-1]

# --- 优化器和学习率配置 ---
OPTIMIZERS = ['Adam', 'SGD']  # 优化器列表
LEARNING_RATES = {
    'Adam': [0.01, 0.001],
    'SGD': [0.01, 0.001]
}  # 每个优化器对应的学习率列表

# --- 输出配置 ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../results/optmizier/1020_optimizer_AdamAndSGD_sinxADDsin5x")

# --- 统计稳定性测试参数 ---
SEED_LIST = [42, 38, 50, 100]  # 使用多个随机种子进行重复实验

# --- 频谱分析工具 ---
# --- 频谱分析工具 ---
N_POLY_ORDER = 100 # 勒让德多项式的阶数
V = orthpoly_coef(None, 'legendre', N_POLY_ORDER)
#x_sample = sin( pi * (-n_poly_order/2 + arange(n_poly_order+1))/(n_poly_order+1) )
x_sample = lege.legroots(1*(np.arange(N_POLY_ORDER+2)==N_POLY_ORDER+1)) # 生成n_poly_order+2个项的勒让德多项式多项式，最高项系数为1，并求解其根
get_fq_coef = lambda f: np.linalg.solve(V, f(x_sample)) # 函数，解线性方程组V * c = f(x_sample)的系数，这组系数就是函数f的频谱，c[0]代表最平缓的成分，c[k]代表越来越高频，变化越来越剧烈的部分


# ==============================================================================
# 2. 模型定义 (Model Definition)
# ==============================================================================
# 简单的全连接神经网络
class SimpleMLP(nn.Module):
    def __init__(self, n_neurons, beta):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, n_neurons),
            nn.Softplus(beta), # default:torch.nn.Softplus(beta=1.0, threshold=20) beta*x   log(1+exp(beta*x)) -> x
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
    y = np.sin(X) + np.sin(5 * X)
    
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

def train_and_evaluate(model, X_train, y_train, X_test, y_test, optimizer_name, lr, epochs, output_dir, seed, results):
    """
    训练模型，评估并绘制损失曲线
    Args:
        model (nn.Module): 模型实例
        X_train (torch.Tensor): 训练集输入
        y_train (torch.Tensor): 训练集标签
        X_test (torch.Tensor): 测试集输入
        y_test (torch.Tensor): 测试集标签
        optimizer_name (str): 优化器名称
        lr (float): 学习率
        epochs (int): 训练轮数
        n (int): 模型神经元数量
        beta (float): Softplus参数
        output_dir (str): 图像输出目录
    Returns:
        std_dev (float): 测试集上的误差标准差
        y_pred (torch.Tensor): 模型在测试集上的预测值
        train_losses (list): 训练损失历史
        val_losses (list): 验证损失历史
    """
    criterion = nn.MSELoss()  # 均方误差损失函数
    
    # 根据optimizer_name选择不同的优化器
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
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

            print(f"Epoch [{epoch}/{epochs}], optimizer={optimizer_name}, lr={lr}, seed={seed} Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

        
        # 如果当前轮次是epochlist中的一个元素
        if epoch + 1 in EPOCHS_LIST:
            # 评估模型在测试集上的性能
            model.eval()    
            with torch.no_grad():
                y_pred = model(X_test)
                error = y_test - y_pred
                std_dev = error.std().item()

                # 定义从 [-1, 1] 到 DATA_RANGE 的变换函数
                a, b = DATA_RANGE[0], DATA_RANGE[1]
                rescale_func = lambda x_norm: (b - a) / 2. * x_norm + (a + b) / 2.
                
                f_ann = lambda x_norm: model(torch.FloatTensor(rescale_func(x_norm)).reshape(-1, 1).to(DEVICE)).cpu().numpy().flatten()
                ann_coeffs = get_fq_coef(f_ann)

                # ====================  !!! 关键修改点 !!! ====================
                # 按照新的结构 'epoch -> optimizer -> lr -> seed' 保存结果
                current_epoch = epoch + 1
                
                # 动态创建嵌套字典
                if optimizer_name not in results['optimizer_results'][current_epoch]:
                    results['optimizer_results'][current_epoch][optimizer_name] = {}
                if lr not in results['optimizer_results'][current_epoch][optimizer_name]:
                    results['optimizer_results'][current_epoch][optimizer_name][lr] = {}
                
                # 存储数据
                results['optimizer_results'][current_epoch][optimizer_name][lr][seed] = {
                    'y_pred': y_pred.cpu().numpy().flatten(),
                    'y_pred_std': std_dev,
                    'coeffs': ann_coeffs
                }

    # --- 绘制并保存当前参数下的损失曲线 ---
    epochs_recorded = np.arange(len(train_losses)) * 10 
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.plot(epochs_recorded, train_losses, label='Train Loss')
    ax.plot(epochs_recorded, val_losses, label='Validation Loss')
    ax.set_title(f'Loss Curve for {optimizer_name}, lr={lr}, seed={seed}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    ax.set_yscale('log')
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_minor_formatter(FormatStrFormatter('%.6g'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.6g'))
    ax.tick_params(axis='y', which='minor', labelsize=8, labelcolor='gray')
    # ax.set_yticks(all_losses)
    # 对y轴的刻度倾斜45度
    # ax.tick_params(axis='y', labelrotation=45)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    # ax.yaxis.set_minor_locator(NullLocator())
    ax.legend()
    ax.grid(True)
    loss_curve_path = os.path.join(output_dir, f'loss_curve_{optimizer_name}_lr{lr}_seed{seed}.png')
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
def main():
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 生成数据
    X_train, y_train, X_test, y_test = generate_data(DATA_RANGE, NUM_POINTS, TEST_SIZE)

    # 计算真实函数的频谱
    # 1. 定义变换
    a, b = DATA_RANGE[0], DATA_RANGE[1]
    rescale_func = lambda x_norm: (b - a) / 2. * x_norm + (a + b) / 2.
    # 2. 定义包装后的真实函数
    true_function_rescaled = lambda x_norm: np.sin(rescale_func(x_norm)) + np.sin(5 * rescale_func(x_norm))
    # 3. 在[-1, 1]上对包装后的函数进行频谱分析
    true_coeffs = get_fq_coef(true_function_rescaled)
    
    # 调整results结构以容纳不同优化器、学习率和epoch的结果
    results = {
        'X_test': X_test.cpu().numpy(),
        'y_test': y_test.cpu().numpy(),
        'X_train': X_train.cpu().numpy(),
        'y_train': y_train.cpu().numpy(),
        'true_coeffs': true_coeffs, # 保存真实频谱
        'fixed_params': {'n_neurons': N_NEURONS, 'beta': BETA, 'epochs': EPOCHS_LIST, 'data_range': DATA_RANGE},
        'optimizer_results': {}
    }
    
    # 只需初始化第一层键 (epoch)
    for epoch_val in EPOCHS_LIST:
        results['optimizer_results'][epoch_val] = {}
    # ===============================================================

    for optimizer_name in OPTIMIZERS:
        optimizer_dir = os.path.join(OUTPUT_DIR, optimizer_name)
        os.makedirs(optimizer_dir, exist_ok=True)
        
        for lr in LEARNING_RATES[optimizer_name]:
            print(f"\n{'='*20} Starting Experiment for {optimizer_name}, lr={lr} {'='*20}")
            
            for seed in SEED_LIST:
                print(f"Setting seed to {seed} for {optimizer_name}, lr={lr}")
                torch.manual_seed(seed)
                np.random.seed(seed)
            
                model = SimpleMLP(n_neurons=N_NEURONS, beta=BETA).to(DEVICE)
                
                std_dev, _ = train_and_evaluate(
                    model, X_train, y_train, X_test, y_test, optimizer_name, lr, epochs, optimizer_dir, seed, results
                )
                print(f"--- Final Result for {optimizer_name}, lr={lr}, seed={seed}: std(y_err) = {std_dev:.6f} ---")

    with open(os.path.join(OUTPUT_DIR, 'optimizer_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()