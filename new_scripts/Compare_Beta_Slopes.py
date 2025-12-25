import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legfit
from sklearn.linear_model import LinearRegression

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 0. Configuration
# ==========================================
# 定义要测试的多个 beta 值
BETA_VALUES = [0.5, 1.0, 2.0, 4.0]
MAX_EPOCHS = 10000  # 固定训练轮次
BINS = 5          # 频谱斜率拟合时的箱子数量
OUTPUTDIR = f"./figures/Compare_Beta_Slopes_bin_{BINS}/" # 保存结果的目录
if not os.path.exists(OUTPUTDIR):
    os.makedirs(OUTPUTDIR)

# ==========================================
# 1. Data Generation
# ==========================================
# We need a dense grid for accurate numerical integration/fitting of Legendre coefficients later.
N_points = 4000 
x_np = np.linspace(-1, 1, N_points)

# Target Function: Sum of sines with distinct frequencies
# Despite having equal amplitude in Fourier basis (sines), 
# they will have a decaying spectrum in Legendre basis.
frequencies = [5, 15, 30, 50, 80]
y_np = np.zeros_like(x_np)
for f in frequencies:
    y_np += 10 * np.sin(f * np.pi * x_np)

# Normalize target to roughly [-1, 1] range for stable training
y_np = y_np / len(frequencies)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(x_np).reshape(-1, 1)
Y_train = torch.FloatTensor(y_np).reshape(-1, 1)

# ==========================================
# 2. Model Definition (MLP with Softplus)
# ==========================================
class MLP(nn.Module):
    def __init__(self, width=1024, beta=1.0):
        super().__init__()
        # Softplus activation with beta parameter
        self.net = nn.Sequential(
            nn.Linear(1, width), nn.Softplus(beta=beta),
            nn.Linear(width, 1)
        )

    def forward(self, x):
        return self.net(x)

# Move data to device if CUDA is available
X_train = X_train.cuda() if torch.cuda.is_available() else X_train
Y_train = Y_train.cuda() if torch.cuda.is_available() else Y_train

# ==========================================
# 3. Utility Functions for Spectral Analysis
# ==========================================
def get_legendre_coeffs(x_grid, y_data, max_degree):
    """
    Computes Legendre coefficients using least squares fitting (legfit).
    Returns the absolute magnitude of coefficients up to max_degree.
    """
    # legfit returns coefficients [c0, c1, c2, ..., c_n]
    coeffs = legfit(x_grid, y_data, max_degree)
    return np.abs(coeffs)


def calculate_decay_base(model, B_singularity, beta):
    """
    计算基于最小 |w_l| 的理论衰减基数 S = 1/Theta_min [cite: 119, 121]。
    
    Args:
        model: MLP 实例
        B_singularity: 激活函数奇点距离 (pi/beta)
        beta: Softplus 激活函数的 beta 参数
        
    Returns:
        min_distance: The smallest distance to a singularity
    """
    # Extract weights and biases from the model 
    layers = list(model.net.children()) 
    
    # Get the last hidden layer (before output layer) 
    first_hidden_layer = layers[0]  # Third Linear layer before output 
    
    # Extract weights and biases 
    w = first_hidden_layer.weight.detach().cpu().numpy().flatten()  # Shape: (width,) 
    b = first_hidden_layer.bias.detach().cpu().numpy()  # Shape: (width,) 
    
    # Calculate singularity distances for each hidden neuron 
    # Formula: (pi/beta - b_i)/w_i 
    distances = np.abs((np.pi / beta) / w) 
    
    # Find the minimum distance 
    if len(distances) == 0: 
        return 0.0, float('inf') 
    
    min_distance = np.min(distances) 
    
    # 使用最小奇点距离作为ratio的分母
    Theta_min = min_distance + np.sqrt(1 + min_distance**2)
    
    return Theta_min, min_distance


def calculate_envelope_slope(coeffs, degrees, n_bins=BINS, threshold=1e-5):
    """
    只针对频谱的“包络线”（局部极大值）进行线性回归，
    获取最精确的收敛率上界。
    """
    # 预处理，去掉极小值
    valid_mask = (coeffs > threshold)
    valid_coeffs = np.log(coeffs[valid_mask])
    valid_degrees = degrees[valid_mask]

    # 对degrees分段
    bins = np.linspace(valid_degrees.min(), valid_degrees.max(), n_bins + 1)
    which_bin = np.digitize(valid_degrees, bins)
    peak_x = []
    peak_y = []

    # 遍历每一个箱子
    for i in range(1, n_bins + 1):
        mask = (which_bin == i)
        if np.any(mask):
            # find the coedffs in the bin
            existing_peaks = valid_coeffs[mask]

            # find the peak in the bin
            peak_idx = np.argmax(existing_peaks)
            peak_x.append(valid_degrees[mask][peak_idx])
            peak_y.append(existing_peaks[peak_idx])
            
    
    # 对局部最大值进行线性拟合
    peak_x = np.array(peak_x).reshape(-1, 1)
    peak_y = np.array(peak_y)
    reg = LinearRegression()
    reg.fit(peak_x, peak_y)
    envelope_slope = -reg.coef_[0]
    r2 = reg.score(peak_x, peak_y)

    return envelope_slope, r2, peak_x, peak_y

# ==========================================
# 4. Main Experiment Loop
# ==========================================
# 存储不同 beta 值的结果
beta_results = {}

# 计算目标函数的频谱系数（只需要计算一次）
max_analyze_degree = int(np.pi * max(frequencies) * 1.2) 
degrees = np.arange(max_analyze_degree + 1)
target_coeffs = get_legendre_coeffs(x_np, y_np, max_analyze_degree)
A_intercept = np.max(target_coeffs)  # 拟合常数 A

# 对每个 beta 值进行训练和分析
for beta in BETA_VALUES:
    print(f"\n=== 开始训练 beta = {beta} ===")
    
    # 创建模型
    model = MLP(beta=beta).cuda() if torch.cuda.is_available() else MLP(beta=beta)
    
    # 训练模型
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{MAX_EPOCHS} | Loss: {loss.item():.8f}")
    
    # 训练完成后计算结果
    print(f"=== beta = {beta} 训练完成 ===")
    
    # 获取最终预测
    output = model(X_train)
    pred_y = output.detach().cpu().numpy().flatten()
    
    # 计算频谱系数
    pred_coeffs = get_legendre_coeffs(x_np, pred_y, max_analyze_degree)
    
    # 计算理想衰减基数和奇点距离
    decay_base, min_singularity_x = calculate_decay_base(model, np.pi / beta, beta)
    
    # 计算实际斜率
    actual_slope, r_squared, peak_x, peak_y = calculate_envelope_slope(pred_coeffs, degrees)
    
    # 存储结果
    beta_results[beta] = {
        'pred_coeffs': pred_coeffs,
        'decay_base': decay_base,
        'min_singularity_x': min_singularity_x,
        'actual_slope': actual_slope,
        'r_squared': r_squared,
        'peak_x': peak_x,
        'peak_y': peak_y
    }

# ==========================================
# 5. Visualization
# ==========================================

# 1. 频谱系数与斜率对比图
plt.figure(figsize=(14, 10))

# 绘制目标函数频谱
plt.scatter(degrees, target_coeffs, s=15, marker='o', color='k', zorder=10, label='Target Function Spectrum')

# 为每个 beta 值绘制频谱和斜率
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 0.9, len(BETA_VALUES))]

for i, beta in enumerate(BETA_VALUES):
    result = beta_results[beta]
    pred_coeffs = result['pred_coeffs']
    decay_base = result['decay_base']
    actual_slope = result['actual_slope']

    
    color = colors[i]
    
    # 绘制实际频谱
    plt.plot(degrees, pred_coeffs, 
             color=color, linewidth=2, alpha=0.8, 
             label=f'Actual Spectrum (beta={beta}, slope={actual_slope:.4f})')
    real_decay = A_intercept * np.exp(-actual_slope * degrees)
    plt.plot(degrees, real_decay, 
            color=color, linewidth=2, alpha=0.5, linestyle=':', 
            label=f'Actual Slope Line (beta={beta})')
              
    # 绘制理想斜率
    theoretical_decay = A_intercept * (decay_base ** (-degrees))
    ideal_k = np.log(decay_base)
    plt.plot(degrees, theoretical_decay, 
             color=color, linewidth=1.5, alpha=0.8, linestyle='--', 
             label=f'Ideal Slope (beta={beta}, S={ideal_k:.4f})')
    plt.scatter(result['peak_x'], np.exp(result['peak_y']), 
                color=color, s=30, marker='x', alpha=0.6)

# 设置图表样式
plt.yscale('log')
plt.xlim(0, max_analyze_degree)
plt.ylim(1e-5, np.max(target_coeffs)*1.5)
plt.title("Spectral Bias and Slope Comparison Across Multiple Beta Values")
plt.xlabel("Legendre Polynomial Degree ($n$)")
plt.ylabel("Coefficient Amplitude $|c_n|$ (Log Scale)")
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTDIR, 'beta_slope_comparison.png'), dpi=300)
plt.show()

# 2. 斜率对比柱状图
plt.figure(figsize=(10, 6))

# 计算理论斜率（从理想衰减基数转换）
theoretical_slopes = [np.log(result['decay_base']) for beta, result in beta_results.items()]
actual_slopes = [result['actual_slope'] for beta, result in beta_results.items()]
betas = list(beta_results.keys())

# 设置柱状图位置
x = np.arange(len(betas))
width = 0.35

# 绘制柱状图
plt.bar(x - width/2, theoretical_slopes, width, label='Theoretical Slope')
plt.bar(x + width/2, actual_slopes, width, label='Actual Slope')

# 设置图表样式
plt.xlabel('Beta Value')
plt.ylabel('Slope')
plt.title('Comparison of Theoretical and Actual Slopes Across Beta Values')
plt.xticks(x, betas)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTDIR, 'beta_slope_bar.png'), dpi=300)
plt.show()

# 3. 奇点距离与斜率关系图
plt.figure(figsize=(10, 6))

min_singularities = [result['min_singularity_x'] for beta, result in beta_results.items()]

# 绘制散点图
plt.scatter(min_singularities, actual_slopes, s=100, color='blue', label='Actual Slope')
plt.scatter(min_singularities, theoretical_slopes, s=100, color='red', marker='x', label='Theoretical Slope')

# 添加回归线
if len(min_singularities) > 1:
    regressor = LinearRegression()
    regressor.fit(np.array(min_singularities).reshape(-1, 1), actual_slopes)
    plt.plot(min_singularities, regressor.predict(np.array(min_singularities).reshape(-1, 1)), color='blue', alpha=0.5)

    regressor2 = LinearRegression()
    regressor2.fit(np.array(min_singularities).reshape(-1, 1), theoretical_slopes)
    plt.plot(min_singularities, regressor2.predict(np.array(min_singularities).reshape(-1, 1)), color='red', alpha=0.5, linestyle='--')

# 设置图表样式
plt.xlabel('Min Singularity Distance $|x_s|$')
plt.ylabel('Slope')
plt.title('Relationship Between Singularity Distance and Slope')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTDIR, 'singularity_slope_relation.png'), dpi=300)
plt.show()

# ==========================================
# 6. Results Summary
# ==========================================
print("\n" + "="*60)
print("实验结果总结")
print("="*60)
print("Beta 值 | 理论斜率 | 实际斜率 | R² 拟合优度 | 最小奇点距离")
print("-"*60)
for beta in BETA_VALUES:
    result = beta_results[beta]
    theoretical_slope = np.log(result['decay_base'])
    print(f"{beta:6.1f} | {theoretical_slope:9.4f} | {result['actual_slope']:8.4f} | {result['r_squared']:13.4f} | {result['min_singularity_x']:14.6f}")

output_summary = os.path.join(OUTPUTDIR, "experiment_summary.txt")
with open(output_summary, "w", encoding="utf-8") as f:
    f.write("="*60 + "\n")
    f.write("实验结果总结\n")
    f.write("="*60 + "\n")
    f.write("Beta 值 | 理论斜率 | 实际斜率 | R² 拟合优度 | 最小奇点距离\n")
    f.write("-"*60 + "\n")
    for beta in BETA_VALUES:
        result = beta_results[beta]
        theoretical_slope = np.log(result['decay_base'])
        f.write(f"{beta:6.1f} | {theoretical_slope:9.4f} | {result['actual_slope']:8.4f} | {result['r_squared']:13.4f} | {result['min_singularity_x']:14.6f}\n")
    f.write("\n训练完成！所有结果已保存为 PNG 文件。\n")
print("\n训练完成！所有结果已保存为 PNG 文件。")
