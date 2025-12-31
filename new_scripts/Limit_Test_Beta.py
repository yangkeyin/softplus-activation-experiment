
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

EPOCHS = 10000
BETA = 4.0  
WIDTH = 100
OUTPUT_PATH = f'./figures/limit_test_beta_sparse_data_epoch{EPOCHS}_beta{BETA}/'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# 检查是否有显卡，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 数据生成 (保持和之前一致)
N_points = 17
x_np = np.linspace(-2*np.pi, 2*np.pi, N_points)
y_np = np.sin(x_np)

x_test_np = np.linspace(-2*np.pi, 2*np.pi, 200)
y_test_ground_truth = np.sin(x_test_np)

f_interp = interp1d(x_np, y_np, kind='linear', fill_value="extrapolate")
y_linear = f_interp(x_test_np)

class MLP(nn.Module):
    def __init__(self, beta=100.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, WIDTH),
            nn.Softplus(beta=beta),
            nn.Linear(WIDTH, 1)
        )
    
    def forward(self, x):
        return self.net(x)
    
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         # 手动调整std：
    #         nn.init.normal_(m.weight, mean=0.0, std=1.0)
    #         nn.init.constant_(m.bias, 0.0)

# 数据准备
X_train = torch.FloatTensor(x_np).reshape(-1, 1).to(device)
Y_train = torch.FloatTensor(y_np).reshape(-1, 1).to(device)
X_test = torch.FloatTensor(x_test_np).reshape(-1, 1).to(device)

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
    model.train()
    opt.zero_grad()
    pred = model(X_train)
    loss = nn.MSELoss()(pred, Y_train)
    loss.backward()
    opt.step()
    scheduler.step()
    loss_history.append(loss.item())
    if e % 1000 == 0:
        print(f"Epoch {e}, Loss: {loss.item():.8f}")

model.eval()
with torch.no_grad():
    y_mlp = model(X_test).cpu().numpy().flatten()

mse_linear = np.mean((y_linear - y_test_ground_truth)**2)
mse_mlp = np.mean((y_mlp - y_test_ground_truth)**2)

plt.figure(figsize=(12, 6))
plt.scatter(x_np, y_np, color='black', label='Training Points (17)', zorder=5)
plt.plot(x_test_np, y_test_ground_truth, 'k-', alpha=0.2, label='Ground Truth (Sine)')
plt.plot(x_test_np, y_linear, 'r--', label=f'Linear Interp (MSE={mse_linear:.6f})')
plt.plot(x_test_np, y_mlp, 'b-', label=f'MLP Beta={BETA} (MSE={mse_mlp:.6f})')

plt.legend()
plt.title(f"Limit Test: Softplus(Beta={BETA}) vs Linear Interpolation")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_PATH, 'comparison.png'))
plt.show()

print(f"Final Result: Linear MSE={mse_linear:.6f}, MLP MSE={mse_mlp:.6f}")
