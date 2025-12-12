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
# 确定 Softplus 的 beta 值
SOFTPLUS_BETA = 2.0 # 使用beta=1.0
# 文献中的奇点距离 B = pi / beta 
B_SINGULARITY = np.pi / SOFTPLUS_BETA

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
    y_np += np.sin(f * np.pi * x_np)

# Normalize target to roughly [-1, 1] range for stable training
y_np = y_np / len(frequencies)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(x_np).reshape(-1, 1)
Y_train = torch.FloatTensor(y_np).reshape(-1, 1)

# ==========================================
# 2. Model Definition (MLP with Softplus)
# ==========================================
class MLP(nn.Module):
    def __init__(self, width=256, beta=1.0):
        super().__init__()
        # Softplus activation with beta parameter
        self.net = nn.Sequential(
            nn.Linear(1, width), nn.Softplus(beta=beta),
            nn.Linear(width, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(beta=SOFTPLUS_BETA).cuda() if torch.cuda.is_available() else MLP(beta=SOFTPLUS_BETA)
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


def calculate_decay_base(model, B_singularity):
    """
    计算基于最小 |w_l| 的理论衰减基数 S = 1/Theta_min [cite: 119, 121]。
    
    Args:
        model: MLP 实例
        B_singularity: 激活函数奇点距离 (pi/beta)
        
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
    distances = np.abs((np.pi / SOFTPLUS_BETA - b) / w) 
    
    # Find the minimum distance 
    if len(distances) == 0: 
        return 0.0, float('inf') 
    
    min_distance = np.min(distances) 
    
    # 使用最小奇点距离作为ratio的分母
    Theta_min = min_distance + np.sqrt(1 + min_distance**2)
    
    
    return Theta_min, min_distance


# ==========================================
# 4. Training Loop with Checkpoints
# ==========================================
optimizer = optim.Adam(model.parameters(), lr=2e-4)
criterion = nn.MSELoss()

# Checkpoints to capture the evolution from early to late training
checkpoints = [200, 1000, 5000, 15000, 30000]
max_epochs = checkpoints[-1]
saved_predictions = {}
# 存储理论衰减基数 S (1/Theta_min)
saved_decay_bases = {}
# 存储最小奇点距离 |x_singularity|
saved_min_singularities_x = {}

print(f"Training with Softplus(beta={SOFTPLUS_BETA}) on device: {X_train.device}")
print(f"Target frequencies: {frequencies}")
print(f"Softplus singularity distance B = pi/beta = {B_SINGULARITY:.4f}")

for epoch in range(1, max_epochs + 1):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, Y_train)
    loss.backward()
    optimizer.step()

    if epoch in checkpoints or epoch == 1:
        # 计算理论衰减基数 S 和最小奇点距离 |x_singularity|
        decay_base, min_singularity_x = calculate_decay_base(model, B_SINGULARITY)
        
        print(f"Epoch {epoch:6d} | Loss: {loss.item():.8f} | Min Singularity |x_s|: {min_singularity_x:.6f} | Decay Base S: {decay_base:.6f}")
        
        saved_predictions[epoch] = output.detach().cpu().numpy().flatten()
        saved_decay_bases[epoch] = decay_base
        saved_min_singularities_x[epoch] = min_singularity_x

# ==========================================
# 5. Spectral Analysis (Legendre Coefficients)
# ==========================================



# Max degree to analyze. Needs to be high enough to capture the highest frequency component.
# Frequency f roughly corresponds to Legendre degree n ~= pi * f
max_analyze_degree = int(np.pi * max(frequencies) * 1.2) 
degrees = np.arange(max_analyze_degree + 1)

# Calculate coefficients for the Ground Truth Target
target_coeffs = get_legendre_coeffs(x_np, y_np, max_analyze_degree)

# ==========================================
# 6. Visualization
# ==========================================
plt.figure(figsize=(14, 8))

# Plot Target Spectrum (Black line)
plt.scatter(degrees, target_coeffs, s=15, marker='o', color='k', zorder=10, label='Target Function Spectrum')

# Prepare colors for checkpoints (using acolormap for time progression)
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 0.9, len(checkpoints))]

# 拟合常数 A (截距)
# 我们需要一个合理的截距 A 来绘制理论斜率线 A * S^k
# 经验上，我们使用最高的系数 |c_0| 作为 A
A_intercept = np.max(target_coeffs)

# Plot Model Predictions and Theoretical Decay Line at Checkpoints
for i, epoch in enumerate(checkpoints):
    pred_y = saved_predictions[epoch]
    pred_coeffs = get_legendre_coeffs(x_np, pred_y, max_analyze_degree)
    decay_base = saved_decay_bases[epoch]
    
    color = colors[i]
    
    # 绘制模型预测的系数曲线
    plt.plot(degrees, pred_coeffs, 
             color=color, linewidth=1.5, alpha=0.8, linestyle='-', 
             label=f'Epoch {epoch}')
              
    # 绘制理论衰减斜率直线 (A * S^k)
    # 在对数-线性图上，这是 log(A) + k * log(S) 的直线
    theoretical_decay = A_intercept * (decay_base ** (-degrees))
    
    # 使用虚线绘制理论斜率
    plt.plot(degrees, theoretical_decay, 
             color=color, linewidth=1.5, alpha=0.8, linestyle='--', 
             label=f'Ideal Slope (S={decay_base:.4f}) at Epoch {epoch}')

# Finalize plot styling
plt.yscale('log')
plt.xlim(0, max_analyze_degree)
# 设置合理的 y 轴范围，以容纳理论斜率线
plt.ylim(1e-7, np.max(target_coeffs)*1.5) 

plt.title(r"Spectral Bias and Theoretical Decay (Softplus, $\beta$={})".format(SOFTPLUS_BETA), fontsize=16)
plt.xlabel("Legendre Polynomial Degree ($n$)", fontsize=14)
plt.ylabel("Coefficient Amplitude $|c_n|$ (Log Scale)", fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.4)

# 重新组织图例，只显示模型曲线和理论斜率的最终点
handles, labels = plt.gca().get_legend_handles_labels()
# 筛选出每对 (Model Line, Ideal Slope) 曲线的图例
# 理想斜率曲线的 label 已经包含 S 值
plt.legend(handles, labels, fontsize=10, loc='lower left', ncol=2)

plt.tight_layout()

print("\nTraining complete. Generating plot with Ideal Slopes...")
plt.show()

# 此外，如果需要，我们可以绘制最小奇点距离随时间的变化。
plt.figure(figsize=(8, 5))
epochs = list(saved_min_singularities_x.keys())
distances = list(saved_min_singularities_x.values())
plt.plot(epochs, distances, marker='o', linestyle='-', color='purple')
plt.title(f"Min Singularity Distance $|x_s|$ vs. Training Epoch ($\beta$={SOFTPLUS_BETA})")
plt.xlabel("Epoch")
plt.ylabel("Min Singularity Distance $|x_s|$")
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.show()
