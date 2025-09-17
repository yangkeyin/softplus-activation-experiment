import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from matplotlib.ticker import ScalarFormatter 

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
N_NEURONS_LIST = [32, 64, 128]  # 您可以修改或增加神经元数量的测试列表
LEARNING_RATE = 0.01
EPOCHS = 1000

# --- 实验核心：β 扫描列表 ---
# 定义当前要使用的 BETA 列表
BETA_TO_RUN = np.arange(5, 51, 5)

# --- 输出配置 ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "experiment_results", f"{BETA_TO_RUN[0]}_{BETA_TO_RUN[-1]}_run1")
# 设置随机种子以保证结果可复现
torch.manual_seed(52)
np.random.seed(52)

# # --- 统计稳定性测试参数 ---
# NUM_RUNS = 5  # 设置每个 (n, β) 组合独立训练的次数

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
    """
    criterion = lambda y_pred, y_true: torch.std(y_true - y_pred)  # 标准差损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []

    best_model_state = None
    best_val_loss = float('inf')
    
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

                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()

            print(f"Epoch [{epoch}/{epochs}], n={n}, beta={beta}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

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
        
    return std_dev


# ==============================================================================
# 5. 主实验流程 (Main Experiment Logic)
# ==============================================================================

if __name__ == "__main__":
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 生成数据
    X_train, y_train, X_test, y_test = generate_data(DATA_RANGE, NUM_POINTS, TEST_SIZE)
    
    # 存储所有实验结果
    results = {}

    # 主循环：遍历神经元数量和beta值
    for n in N_NEURONS_LIST:
        print(f"\n{'='*20} Starting Experiment for n = {n} Neurons {'='*20}")
        results[n] = {'betas': [], 'stds': []}
        
        for beta in BETA_TO_RUN:
            print(f"\n--- Training with n={n}, beta={beta} ---")
            
            # 实例化模型
            model = SimpleMLP(n_neurons=n, beta=beta).to(DEVICE)
            
            # 训练和评估
            std_dev = train_and_evaluate(
                model, X_train, y_train, X_test, y_test, LEARNING_RATE, EPOCHS, n, beta, OUTPUT_DIR
            )
            
            # 记录结果
            results[n]['betas'].append(beta) # x
            results[n]['stds'].append(std_dev) # y
            
            print(f"--- Result for n={n}, beta={beta}: std(y_err) = {std_dev:.6f} ---")

    # --- 绘制并保存最终的 std(y_err) vs. beta 总结图 ---
    print("\nAll experiments finished. Generating final summary plot...")
    plt.figure(figsize=(12, 8))
    
    for n, data in results.items():
        # 将结果排序，以便绘图
        sorted_data = sorted(zip(data['betas'], data['stds'])) # 先配对成元组，再对第一个元素进行排序， 防止输入的是无序的
        betas, stds = zip(*sorted_data) # 解压缩
        plt.plot(betas, stds, marker='o', linestyle='-', label=f'n = {n} neurons') # 将所有线绘制在同一张图上

    plt.xscale('log')  # Beta值跨度很大，使用对数坐标轴更清晰
    plt.title('Standard Deviation of Error vs. Beta (Smoothness Parameter)')
    plt.xlabel('Beta (β) - Log Scale')
    plt.ylabel('Standard Deviation of Error (std(y_err))')
    plt.legend()
    plt.grid(True, which="major", ls="--")
    ax = plt.gca() # 获取当前坐标轴
    ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.xticks(BETA_TO_RUN, labels=[str(b) for b in BETA_TO_RUN], rotation=45, ha='right')  #它会在X轴上，精确地在每个 Beta 值的位置上创建带标签的刻度。
    
    summary_plot_path = os.path.join(OUTPUT_DIR, 'summary_std_vs_beta.png')
    plt.savefig(summary_plot_path)
    print(f"Final summary plot saved to: {summary_plot_path}")
    plt.show()