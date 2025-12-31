
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
EPOCHS = 10000
BETA = 100.0  
OUTPUT_PATH = f'./figures/limit_test_beta_epoch{EPOCHS}_beta{BETA}_4000duan/'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# 检查是否有显卡，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 数据生成 (保持和之前一致)
N_points = 4000
x_np = np.linspace(-1, 1, N_points)
frequencies = [5, 15, 30, 50, 80]
y_np = np.zeros_like(x_np)
for f in frequencies: y_np += np.sin(f * np.pi * x_np)
y_np = y_np / len(frequencies)

# 2. 分段线性插值基准 (使用隐藏层宽度作为节点数)

f_interp = interp1d(x_np, y_np, kind='linear', fill_value="extrapolate")
y_linear = f_interp(x_np)
mse_linear = np.mean((y_linear - y_np)**2)

# 3. 训练极限 Beta 模型 (Beta=100)
class MLP(nn.Module):
    def __init__(self, beta=100.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 1024), nn.Softplus(beta=beta), nn.Linear(1024, 1))
    def forward(self, x): return self.net(x)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 手动调整std：
            nn.init.normal_(m.weight, mean=0.0, std=1.0)
            nn.init.constant_(m.bias, 0.0)



X = torch.FloatTensor(x_np).reshape(-1, 1).to(device)
Y = torch.FloatTensor(y_np).reshape(-1, 1).to(device)
model = MLP(beta=BETA).to(device)
print(f"Before INIT STD: {model.net[0].weight.std().item():.4f}")
# model.apply(model._init_weights)
# print(f"After INIT STD: {model.net[0].weight.std().item():.4f}")
std_init = model.net[0].weight.std().item()
opt = optim.Adam(model.parameters(), lr=1e-3)
# 调整学习率
scheduler = optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)
print( "Training MLP with Beta=4...")
loss_history = []
for e in range(EPOCHS):
    opt.zero_grad(); loss = nn.MSELoss()(model(X), Y); loss.backward(); opt.step()
    scheduler.step()
    loss_history.append(loss.item())
    if e % 10 == 0: 
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {e}, Loss: {loss.item():.6f}, LR: {current_lr:.6f}")

y_mlp = model(X).detach().cpu().numpy().flatten()
mse_mlp = np.mean((y_mlp - y_np)**2)

# 绘制loss曲线
plt.figure(figsize=(10, 4))
plt.plot(loss_history, label='Training Loss')
plt.yscale( 'log' )
plt.title("Loss Curve (Log Scale)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True, which="both", ls="-")
loss_path = os.path.join(OUTPUT_PATH, 'loss_curve.png')
plt.savefig(loss_path)
plt.show()
plt.close()
print(f"Loss curve saved: {loss_path}")

# 4. 绘图对比
plt.figure(figsize=(12, 6))
plt.plot(x_np, y_np, 'k', alpha=0.3, label='Target')
plt.plot(x_np, y_linear, 'r--', label=f'Linear Interp (MSE={mse_linear:.6f})')
plt.plot(x_np, y_mlp, 'b:', label=f'MLP Beta={BETA} (MSE={mse_mlp:.6f}, std={std_init:.4f})')
plt.legend()
plt.title(f"Limit Test: MLP (Beta={BETA}) vs Piecewise Linear Interpolation")
result_path = os.path.join(OUTPUT_PATH, 'limit_test_comparison.png')
plt.savefig(result_path)
plt.show()
plt.close()
print(f"Comparison plot saved: {result_path}")
print(f"Result: Linear MSE={mse_linear:.6f}, MLP MSE={mse_mlp:.6f}")