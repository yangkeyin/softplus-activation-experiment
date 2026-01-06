# new_scripts/Limit_Test_Std_Analysis.py

import os
import argparse
import numpy as np
import torch
# 先把原本的引用拿过来，之后用到再补
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
import json
from scipy.interpolate import interp1d
from numpy.polynomial.legendre import legfit


# --- 工程师思维：把所有可能变的量，都变成参数 ---
def parse_args():
    parser = argparse.ArgumentParser(description="分析网络深度(Layers)对频谱特性的影响")
    
    # 1. 核心实验变量 (我们要研究的对象)
    parser.add_argument('--train_points', type=int, default=200, help='训练点数 (建议设为 < width)')
    parser.add_argument('--init_std', type=float, default=None, help='权重初始化的标准差')
    
    # 2. 也是变量，但这次实验可能固定住
    parser.add_argument('--width', type=int, default=1024, help='神经网络宽度')
    parser.add_argument('--beta', type=float, default=4.0, help='Softplus Beta')
    parser.add_argument('--epochs', type=int, default=10000, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--degree', type=int, default=300, help='Legendre 多项式次数')
    parser.add_argument('--test_points', type=int, default=4000, help='测试点数量')
    # 新增layers变量
    parser.add_argument('--num_layers', type=int, default=2, help='神经网络层数')
    
    # 3. 工程化变量 (为了保存结果不混乱)
    parser.add_argument('--save_dir', type=str, default='../figuers', help='结果保存的根目录')
    parser.add_argument('--exp_name', type=str, default='std_analysis', help='实验名称标签')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，保证可复现')
    
    return parser.parse_args()

class MLP(nn.Module):
    # 这里的 init_std 是我们新加的参数
    def __init__(self, width, beta, init_std = None, num_layers=2): # 新增layers参数
        super().__init__()
        # 动态构建层
        layers = []

        # 1. 输入层 (Input -> Hidden)
        layers.append(nn.Linear(1, width))
        layers.append(nn.Softplus(beta=beta))

        # 2. 中间层循环 (Hidden -> Hidden)
        # 总层数=num_layers，减去头尾各1层，中间有 num_layers-2 层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Softplus(beta=beta))

        # 3. 输出层 (Hidden -> Output)
        # 注意：最后一层通常不加激活函数，或者根据任务定。这里拟合任务直接输出。
        layers.append(nn.Linear(width, 1))

        self.net = nn.Sequential(*layers)
        
        if init_std is not None:
            self.init_std = init_std
            # 关键一步：应用自定义初始化
            self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 工程师思维：这里不能写死 1.0，要用 self.init_std
            nn.init.normal_(m.weight, mean=0.0, std=self.init_std)
            nn.init.constant_(m.bias, 0.0)
            
    def forward(self, x):
        return self.net(x)
def get_legendre_coeffs(x_grid, y_data, degree):
    """
    args:
        x_grid: 输入数据的网格点，形状为 (train_points,)
        y_data: 对应于 x_grid 的目标值，形状为 (train_points,)
        max_degree: 最大的 Legendre 多项式次数
        
    return:
        coeffs: 形状为 (max_degree+1,) 的 Legendre 多项式系数数组
    """
    coeffs = legfit(x_grid, y_data, degree)
    return np.abs(coeffs)

def get_y(x, frequencies):
    y = np.zeros_like(x)
    for f in frequencies:
        y += 10 * np.sin(f * np.pi * x)
    y = y / len(frequencies)
    return y

def main():
    args = parse_args()
    
    # 1. 搞定随机性 (Replicability)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 2. 搞定保存路径 (这是很多新手忽略的，导致结果找不到)
    # 我们用 时间戳 + 关键参数 命名文件夹
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{args.exp_name}_Layers{args.num_layers}_Beta{args.beta}_N{args.train_points}_Width{args.width}"
    output_path = os.path.join(args.save_dir, run_id)
    os.makedirs(output_path, exist_ok=True)
    
    # 把参数存下来！以后你写论文要回溯的。
    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    # 3. 准备数据 (用 args.train_points)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 训练数据：稀疏的点
    x_np = np.linspace(-1, 1, args.train_points)
    frequencies = [5, 15, 30, 50, 80]
    y_np = get_y(x_np, frequencies)
    
    # 测试数据：密集的点 (Ground Truth)
    x_test_np = np.linspace(-1, 1, args.train_points * 10)
    y_test_ground_truth = get_y(x_test_np, frequencies)

    # 线性插值
    f_interp = interp1d(x_np, y_np, kind='linear', fill_value="extrapolate")
    y_linear = f_interp(x_test_np)  
        
    # 转换为 Tensor
    X_train = torch.FloatTensor(x_np).reshape(-1, 1).to(device)
    Y_train = torch.FloatTensor(y_np).reshape(-1, 1).to(device)
    X_test = torch.FloatTensor(x_test_np).reshape(-1, 1).to(device)
    
    # 4. 初始化模型 (传入 args)
    model = MLP(width=args.width, beta=args.beta, init_std=args.init_std, num_layers=args.num_layers).to(device)
    # [新增] 工程师的强迫症：打印出来数一数，是不是真的是 args.layers 层？
    print("\nModel Structure Check:")
    print(model)
    print(f"Total Parameter Count: {sum(p.numel() for p in model.parameters())}")
    print("-" * 30)
    # 获取模型的std
    with torch.no_grad():
        init_std = model.net[0].weight.std().item()
    
    # 5. 训练循环 (和原来一样，但加上简单的进度打印)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)
    
    print(f"Start Training: {run_id}")
    for e in range(args.epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_train)
        loss = nn.MSELoss()(pred, Y_train)
        loss.backward()
        opt.step()
        scheduler.step()
        
        if e % 2000 == 0:
            print(f"Epoch {e}, Loss: {loss.item():.6f}")
            
    # 6. 绘图与保存 (Visualisation)
    model.eval()
    with torch.no_grad():
        y_mlp = model(X_test).cpu().numpy().flatten()

    # 计算 MSE
    mse_linear = np.mean((y_linear - y_test_ground_truth)**2)
    mse_mlp = np.mean((y_mlp - y_test_ground_truth)**2)

    # 获得频谱系数
    degree = args.degree
    # 计算 MLP 的频谱
    mlp_coeffs = get_legendre_coeffs(x_test_np, y_mlp, degree=degree)
    # 计算 Ground Truth 的频谱（作为对照组）
    gt_coeffs = get_legendre_coeffs(x_test_np, y_test_ground_truth, degree=degree)
    # 计算线性插值的频谱
    linear_coeffs = get_legendre_coeffs(x_test_np, y_linear, degree=degree)


    # 绘制频谱系数图
    plt.figure(figsize=(10, 5))
    # 使用 log scale 往往更能看出高频衰减的差异
    plt.scatter(range(degree + 1), gt_coeffs, color='k', alpha=0.8, label='Ground Truth Spectrum')
    # plt.scatter(range(degree + 1), linear_coeffs, color='r', label=f'Linear Interp Spectrum')
    plt.semilogy(range(degree + 1), mlp_coeffs, 'b-o', markersize=4, label=f'MLP Spectrum')

    plt.ylim(1e-5, max(gt_coeffs) * 1.5)
    plt.xlabel('Legendre Polynomial Degree (Frequency Indicator)')
    plt.ylabel('Coefficient Magnitude (Log Scale)')
    plt.title(f"Spectral Analysis: Layers={args.num_layers}, Parameters={sum(p.numel() for p in model.parameters())}")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.savefig(os.path.join(output_path, 'spectrum_analysis.png'))
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(x_test_np, y_test_ground_truth, 'k-', alpha=0.6, label='Target')
    # plt.plot(x_test_np, y_linear, 'r--', label=f'Linear Interp (MSE={mse_linear:.6f})')
    plt.plot(x_test_np, y_mlp, 'b-', label=f'MLP Beta={args.beta} (MSE={mse_mlp:.6f})')
    plt.title(f"Fit Analysis: Layers={args.num_layers}, Parameters={sum(p.numel() for p in model.parameters())}")   
    plt.legend()
    plt.savefig(os.path.join(output_path, 'comparison.png'))
    plt.show()
    plt.close()

    save_data = {
        "x_test": x_test_np,
        "y_ground_truth": y_test_ground_truth,
        "y_mlp": y_mlp,
        "mlp_coeffs": mlp_coeffs,
        "gt_coeffs": gt_coeffs,
        "mse_mlp": mse_mlp,
        'args' : vars(args)

    }

    # 保存为 numpy格式
    np.save(os.path.join(output_path, 'results.npy'), save_data)
    print(f"Raw results saved to {os.path.join(output_path, 'results.npy')}")
    print(f"Final Result: Linear MSE={mse_linear:.6f}, MLP MSE={mse_mlp:.6f}")
    print(f"Experiment Finished. Saved to {output_path}")

if __name__ == "__main__":
    main()