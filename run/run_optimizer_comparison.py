import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import pickle
from matplotlib.ticker import FormatStrFormatter, NullLocator, MaxNLocator

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
EPOCHS = 1000  # 固定为1000轮次

# --- 优化器和学习率配置 ---
OPTIMIZERS = ['Adam', 'SGD']  # 优化器列表
LEARNING_RATES = {
    'Adam': [0.01, 0.001],
    'SGD': [0.01, 0.001]
}  # 每个优化器对应的学习率列表

# --- 输出配置 ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "new_optimizer_EPOCH1000")

# --- 统计稳定性测试参数 ---
SEED_LIST = [42, 38, 50, 100]  # 使用多个随机种子进行重复实验

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

def train_and_evaluate(model, X_train, y_train, X_test, y_test, optimizer_name, lr, epochs, n, beta, output_dir):
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
    # 合并训练损失和验证损失，去除重复值并排序，以此设置 y 轴刻度
    all_losses = sorted(set(train_losses + val_losses))
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
        
    return std_dev, y_pred, train_losses, val_losses


# ==============================================================================
# 5. 主实验流程 (Main Experiment Logic)
# ==============================================================================
def main():
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 生成数据
    X_train, y_train, X_test, y_test = generate_data(DATA_RANGE, NUM_POINTS, TEST_SIZE)
    
    # 调整results结构以容纳不同优化器和学习率的结果
    results = {
        # 直接将测试数据存进去 (注意从Tensor转为Numpy array)
        'X_test': X_test.cpu().numpy(),
        'y_test': y_test.cpu().numpy(),
        'X_train': X_train.cpu().numpy(),
        'y_train': y_train.cpu().numpy(),
        'fixed_params': {
            'n_neurons': N_NEURONS,
            'beta': BETA,
            'epochs': EPOCHS
        },
        # 存储不同优化器和学习率的结果
        'optimizer_results': {}
    }
    
    # 为每个优化器创建子目录
    for optimizer_name in OPTIMIZERS:
        optimizer_dir = os.path.join(OUTPUT_DIR, optimizer_name)
        if not os.path.exists(optimizer_dir):
            os.makedirs(optimizer_dir)
        
        results['optimizer_results'][optimizer_name] = {}
        
        # 遍历当前优化器的所有学习率
        for lr in LEARNING_RATES[optimizer_name]:
            print(f"\n{'='*20} Starting Experiment for {optimizer_name}, lr={lr} {'='*20}")
            
            results['optimizer_results'][optimizer_name][lr] = {}
            
            for seed in SEED_LIST:
                # 设置随机种子
                print(f"Setting seed to {seed} for {optimizer_name}, lr={lr}")
                torch.manual_seed(seed)
                np.random.seed(seed)
            
                # 实例化模型
                model = SimpleMLP(n_neurons=N_NEURONS, beta=BETA).to(DEVICE)
                
                # 训练和评估
                std_dev, y_pred, train_losses, val_losses = train_and_evaluate(
                    model, X_train, y_train, X_test, y_test, optimizer_name, lr, EPOCHS, N_NEURONS, BETA, optimizer_dir
                )

                # 存储结果
                results['optimizer_results'][optimizer_name][lr][seed] = {
                    'y_pred_std': std_dev,
                    'y_pred': y_pred.cpu().numpy().flatten(),
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }

                print(f"--- Result for {optimizer_name}, lr={lr}, seed={seed}: std(y_err) = {std_dev:.6f} ---")

    # 将results保存为pickle文件
    with open(os.path.join(OUTPUT_DIR, 'optimizer_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()