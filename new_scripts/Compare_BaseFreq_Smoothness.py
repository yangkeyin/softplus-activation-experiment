import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.polynomial.legendre import legfit
from sklearn.linear_model import LinearRegression
import os

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 0. Configuration
# ==========================================
# 定义要测试的 Base Frequency 值 (跨度要大，模拟不同尺度的平滑偏好)
# 猜想：Base越小 -> 包含更高频分量 -> 拟合越锐利 -> 频谱衰减越慢
BASE_FREQ_VALUES = [10.0, 500.0, 10000.0, 1000000.0] 
MAX_EPOCHS = 100  # Transformer 拟合通常比 MLP 快，但需要足够轮次收敛
D_MODEL = 64      # 隐藏层维度
N_HEAD = 2         # 多头注意力
OUTPUT_DIR = "./figures/Compare_BaseFreq_Smoothness_d64_epoch100_head2/" # 保存结果的目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 1. Data Generation (Copy from reference)
# ==========================================
N_points = 4000 
x_np = np.linspace(-1, 1, N_points)

# Target Function: Sum of sines with distinct frequencies
frequencies = [5, 15, 30, 50, 80] # 略微降低最高频，适配Transformer的学习难度
y_np = np.zeros_like(x_np)
for f in frequencies:
    y_np += np.sin(f * np.pi * x_np)

# Normalize target
y_np = y_np / len(frequencies)

# Convert to PyTorch tensors (Batch first for Transformer: [Batch, Seq, Dim])
# 这里我们将整个序列作为一个 Batch 处理
X_train = torch.FloatTensor(x_np).reshape(1, -1, 1) # [1, 2000, 1]
Y_train = torch.FloatTensor(y_np).reshape(1, -1, 1) # [1, 2000, 1]

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = X_train.to(device)
Y_train = Y_train.to(device)

# ==========================================
# 2. Model Definition (Transformer with Tunable PE)
# ==========================================
class TunablePE(nn.Module):
    def __init__(self, d_model, max_len=5000, base=10000.0):
        super().__init__()
        self.base = base
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 核心：基频控制波长分布
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(base) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SimpleTransformerRegressor(nn.Module):
    def __init__(self, d_model=128, n_head=4, base_freq=10000.0):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = TunablePE(d_model, max_len=N_points, base=base_freq)
        
        # 使用简单的 Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2) # 2层足以观察平滑性
        
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # 捕获 Attention map 需要 hook 或改写 layer，这里为了简化，
        # 我们只关注输出的频谱特性，近似认为输出特性反映了内部处理
        x = self.transformer_encoder(x)
        return self.output_proj(x)

    def get_internal_smoothness_metrics(self, x):
        """辅助函数：计算内部特征的狄利克雷能量，作为辅助验证"""
        x_emb = self.input_proj(x)
        x_pe = self.pos_encoder(x_emb)
        features = self.transformer_encoder(x_pe) # [1, Seq, Dim]
        
        # 计算 Dirichlet Energy (特征差异性)
        features_norm = features / (torch.norm(features, p=2, dim=-1, keepdim=True) + 1e-9)
        diff = features_norm[:, 1:, :] - features_norm[:, :-1, :]
        energy = torch.mean(torch.norm(diff, p=2, dim=-1)**2)
        
        return energy.item()

# ==========================================
# 3. Utility Functions (Spectral Analysis)
# ==========================================
def get_legendre_coeffs(x_grid, y_data, max_degree):
    coeffs = legfit(x_grid, y_data, max_degree)
    return np.abs(coeffs)

def calculate_actual_slope(coeffs, degrees, threshold=1e-6):
    valid_indices = np.where(coeffs > threshold)[0]
    if len(valid_indices) < 5: return 0.0, 0.0
    
    valid_degrees = degrees[valid_indices]
    valid_log_coeffs = np.log(coeffs[valid_indices])
    
    regressor = LinearRegression()
    regressor.fit(valid_degrees.reshape(-1, 1), valid_log_coeffs)
    return -regressor.coef_[0], regressor.score(valid_degrees.reshape(-1, 1), valid_log_coeffs)

# ==========================================
# 4. Main Experiment Loop
# ==========================================
results = {}

# 计算目标频谱
max_analyze_degree =  int(np.pi * max(frequencies)) # 足够覆盖目标频率
degrees = np.arange(max_analyze_degree + 1)
target_coeffs = get_legendre_coeffs(x_np, y_np, max_analyze_degree)

for base in BASE_FREQ_VALUES:
    print(f"\n=== Training Base Freq = {base} ===")
    
    model = SimpleTransformerRegressor(d_model=D_MODEL, n_head=N_HEAD, base_freq=base).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    loss_history = []
    
    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")
        loss_history.append(loss.item())
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        output = model(X_train)
        pred_y = output.cpu().numpy().flatten()
        
        # 1. 频谱分析
        pred_coeffs = get_legendre_coeffs(x_np, pred_y, max_analyze_degree)
        slope, r2 = calculate_actual_slope(pred_coeffs, degrees)
        
        # 2. 内部光滑性分析 (Dirichlet Energy)
        # Energy 越大 -> 特征越尖锐 -> 越不平滑
        energy = model.get_internal_smoothness_metrics(X_train)
    
    results[base] = {
        'pred_y': pred_y,
        'pred_coeffs': pred_coeffs,
        'slope': slope,
        'energy': energy,
        'final_loss': loss_history[-1]
    }
    print(f"-> Slope: {slope:.4f} | Energy: {energy:.4f}")

# ==========================================
# 5. Visualization (Mimicking Reference)
# ==========================================

# --- Plot 1: 频谱衰减对比 (Spectral Decay) ---
plt.figure(figsize=(14, 8))
plt.scatter(degrees, target_coeffs, s=20, color='k', label='Target Spectrum', zorder=10)

cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 0.9, len(BASE_FREQ_VALUES))]

for i, base in enumerate(BASE_FREQ_VALUES):
    res = results[base]
    label = f"Base={base:.0f}"
    plt.plot(degrees, res['pred_coeffs'], color=colors[i], linewidth=2, alpha=0.8, label=label)

plt.yscale('log')
plt.title("Spectral Bias: Effect of Position Encoding Base Frequency")
plt.xlabel("Legendre Degree (Frequency)")
plt.ylabel("Coefficient Magnitude (Log Scale)")
plt.ylim(1e-5, 1.0)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "BaseFreq_Spectral_Decay.png")
plt.show()

# --- Plot 2: 拟合曲线对比 (Spatial Domain) ---
plt.figure(figsize=(14, 6))
plt.plot(x_np, y_np, 'k--', alpha=0.4, label='Target', linewidth=1)

for i, base in enumerate(BASE_FREQ_VALUES):
    plt.plot(x_np, results[base]['pred_y'], color=colors[i], linewidth=1.5, alpha=0.8, 
             label=f"Base={base:.0f} (Loss={results[base]['final_loss']:.1e})")

plt.title("Spatial Fitting Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "BaseFreq_Fitting_Comparison.png")
plt.show()

# --- Plot 3: 核心指标关联 (Metrics Correlation) ---
plt.figure(figsize=(10, 5))

# 子图1: Base Freq vs Slope (越低越平滑?)
plt.subplot(1, 2, 1)
slopes = [results[b]['slope'] for b in BASE_FREQ_VALUES]
bases_log = np.log10(BASE_FREQ_VALUES)

plt.plot(bases_log, slopes, 'o-', linewidth=2, markersize=8)
plt.xlabel("Log10(Base Frequency)")
plt.ylabel("Spectral Decay Slope (Lower is Slower Decay)")
plt.title("Freq Bias vs. Base Frequency")
plt.grid(True)
plt.gca().invert_yaxis() # 注意：Slope越小表示衰减越慢(越高频)，反转Y轴便于直观理解"高频能力"

# 子图2: Dirichlet Energy (Internal Smoothness)
plt.subplot(1, 2, 2)
energies = [results[b]['energy'] for b in BASE_FREQ_VALUES]

plt.bar(range(len(BASE_FREQ_VALUES)), energies, color=colors)
plt.xticks(range(len(BASE_FREQ_VALUES)), [str(int(b)) for b in BASE_FREQ_VALUES])
plt.xlabel("Base Frequency")
plt.ylabel("Dirichlet Energy (Higher is Sharper)")
plt.title("Internal Feature Sharpness")
plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + "BaseFreq_Metrics_Correlation.png")
plt.show()

# ==========================================
# 6. Summary
# ==========================================
print("\n" + "="*60)
print("实验结果总结: Base Frequency 对 Transformer 光滑性的影响")
print("="*60)
print(f"{'Base Freq':<12} | {'Slope (Spec Decay)':<20} | {'Energy (Sharpness)':<20} | {'MSE Loss':<12}")
print("-" * 70)
for base in BASE_FREQ_VALUES:
    res = results[base]
    # Slope: 值越大，衰减越快（越平滑）。值越小，保留高频越多。
    # Energy: 值越大，特征越尖锐。值越小，Over-smoothing。
    print(f"{base:<12.1f} | {res['slope']:<20.4f} | {res['energy']:<20.4f} | {res['final_loss']:.5f}")