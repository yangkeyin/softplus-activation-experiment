import numpy as np
import torch
import torch.nn as nn
import os
import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import rescale, get_fq_coef

# 配置参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FINETUNE_EPOCHS = 1000
FINETUNE_LR = 0.0001
FINETUNE_EVAL_STEP = 10

# 输出目录配置
OUTPUT_DIR = "figures/beta_finetune"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 异常点情景配置
OUTLIER_SCENARIOS = {
    'Scenario_1_Single': [(0.0, 5.0)],
    'Scenario_2_Multiple': [(0.0, 5.0), (np.pi/2, -3.0), (-np.pi, 2.0)]
}


class FNNModel(nn.Module):
    def __init__(self, n, beta):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, n),
            nn.Softplus(beta=beta),
            nn.Linear(n, 1)
        )
    
    def forward(self, x):
        return self.layers(x)


def set_seed(seed_value):
    """设置所有需要随机种子的库的种子。"""
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    import random
    random.seed(seed_value)


def prepare_finetune_data(x_train_base, y_train_base, outlier_list, device):
    """准备微调数据，包括原始训练数据和异常点"""
    # 转换异常点为张量
    x_outliers = torch.tensor([[x] for x, _ in outlier_list], dtype=torch.float32).to(device)
    y_outliers = torch.tensor([[y] for _, y in outlier_list], dtype=torch.float32).to(device)
    
    # 合并原始训练数据和异常点
    x_train_ft = torch.cat([x_train_base, x_outliers], dim=0)
    y_train_ft = torch.cat([y_train_base, y_outliers], dim=0)
    
    # 保存用于绘图的散点数据
    x_train_ft_scatter = x_train_ft.cpu().numpy()
    y_train_ft_scatter = y_train_ft.cpu().numpy()
    
    return x_train_ft, y_train_ft, x_train_ft_scatter, y_train_ft_scatter


def calculate_metrics(model, x_train_ft, y_train_ft, y_train, x_test, y_test, y_pred_test_base, 
                     outlier_list, true_coef, normalized_pred_func):
    """计算微调后的所有指标"""
    model.eval()
    with torch.no_grad():
        # 基础预测
        y_pred_train_ft = model(x_train_ft)
        y_pred_test_ft = model(x_test)
        
        # 计算基础RMSE - 使用包含异常点的训练数据目标值
        train_rms_ft = torch.sqrt(torch.mean((y_pred_train_ft - y_train_ft)**2)).item()
        test_rms_ft = torch.sqrt(torch.mean((y_pred_test_ft - y_test)**2)).item()
        
        # 计算微调影响扭曲度
        y_distortion_rms = torch.sqrt(torch.mean((y_pred_test_ft - y_pred_test_base)**2)).item()
        
        # 计算频谱误差
        pred_coef = get_fq_coef(normalized_pred_func)
        spectrum_error_ft = np.sqrt(np.mean((pred_coef - true_coef)** 2))
        
        # 计算局部拟合误差
        local_errors = []
        for x, y in outlier_list:
            x_tensor = torch.tensor([[x]], dtype=torch.float32).to(DEVICE)
            y_pred = model(x_tensor)
            local_errors.append(torch.abs(y_pred - y).item())
        local_fit_error = np.mean(local_errors)
        
        # 计算全局损伤（远场区域）
        # 选择远场区域：|x| > pi
        far_field_mask = torch.abs(x_test) > np.pi
        if torch.any(far_field_mask):
            x_far = x_test[far_field_mask]
            y_far_true = y_test[far_field_mask]
            y_far_pred = y_pred_test_ft[far_field_mask]
            global_damage_rms = torch.sqrt(torch.mean((y_far_pred - y_far_true)**2)).item()
        else:
            global_damage_rms = 0.0
    
    return {
        # 基础指标
        "y_pred_test_ft": y_pred_test_ft.cpu().detach().numpy().flatten(),
        "y_pred_train_ft": y_pred_train_ft.cpu().detach().numpy().flatten(),
        # 数值指标
        "train_rms_ft": train_rms_ft,
        "test_rms_ft": test_rms_ft,
        "y_distortion_rms": y_distortion_rms,
        "spectrum_error_ft": spectrum_error_ft,
        "local_fit_error": local_fit_error,
        "global_damage_rms": global_damage_rms,
        "pred_coef_ft": pred_coef
    }


def main():
    # 加载基线数据
    baseline_file = "figures/beta_base/results_base.pkl"
    if not os.path.exists(baseline_file):
        print(f"错误：找不到基线结果文件 {baseline_file}")
        print("请先运行 run_beta.py 生成基线结果")
        return
    
    print(f"加载基线结果文件：{baseline_file}")
    with open(baseline_file, 'rb') as f:
        results_base = pickle.load(f)
    
    # 提取必要的参数和数据
    BETA = results_base['BETA']
    SEEDS = results_base['SEEDS']
    BASELINE_EPOCH = results_base['BASELINE_EPOCH']
    true_coef = results_base['true_coef']
    
    # 转换数据到张量并移动到设备
    x_train = torch.tensor(results_base['x_train'], dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(results_base['y_train'], dtype=torch.float32).to(DEVICE)
    x_test = torch.tensor(results_base['x_test'], dtype=torch.float32).to(DEVICE)
    y_test = torch.tensor(results_base['y_test'], dtype=torch.float32).to(DEVICE)
    
    # 初始化微调结果字典
    finetune_results = {beta: {seed: {} for seed in SEEDS} for beta in BETA}
    
    # 外层循环：遍历每个beta值
    for beta in BETA:
        print(f"\n处理 beta={beta}")
        
        # 中层循环：遍历每个种子
        for seed in SEEDS:
            print(f"  处理 seed={seed}")
            
            # 加载预训练模型
            model_path = f"figures/beta_base/models/model_beta_{beta}_seed_{seed}_epoch_{BASELINE_EPOCH}.pth"
            if not os.path.exists(model_path):
                print(f"    警告：找不到模型文件 {model_path}")
                continue
            
            # 初始化模型
            model = FNNModel(n=100, beta=beta)
            model.to(DEVICE)
            model.load_state_dict(torch.load(model_path))
            
            # 获取基线预测结果
            baseline_metrics = results_base['metrics'][beta][seed][BASELINE_EPOCH]
            y_pred_test_base = torch.tensor(baseline_metrics['y_pred_test_base'], dtype=torch.float32).to(DEVICE)
            
            # 为模型创建归一化预测函数
            normalized_pred_func = lambda x: model(torch.tensor(rescale(x, [-2*np.pi, 2*np.pi]), 
                                                           dtype=torch.float32).reshape(-1, 1).to(DEVICE)).cpu().detach().numpy().flatten()
            
            # 内层循环：遍历每个异常点情景
            for scenario_name, outlier_list in OUTLIER_SCENARIOS.items():
                print(f"    处理情景: {scenario_name}")
                finetune_results[beta][seed][scenario_name] = {}
                
                # 准备微调数据
                x_train_ft, y_train_ft, x_train_ft_scatter, y_train_ft_scatter = \
                    prepare_finetune_data(x_train, y_train, outlier_list, DEVICE)
                
                # 创建新的优化器
                optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)
                criterion = nn.MSELoss()
                
                # 微调循环
                for epoch in range(MAX_FINETUNE_EPOCHS):
                    # 训练模式
                    model.train()
                    optimizer.zero_grad()
                    
                    # 前向传播
                    y_pred_ft = model(x_train_ft)
                    loss = criterion(y_pred_ft, y_train_ft)
                    
                    # 反向传播和优化
                    loss.backward()
                    optimizer.step()
                    
                    # 评估和保存结果
                    if (epoch + 1) % FINETUNE_EVAL_STEP == 0:
                        print(f"      微调 epoch {epoch+1}/{MAX_FINETUNE_EPOCHS}, Loss: {loss.item():.6f}")
                        
                        # 计算所有指标
                        metrics = calculate_metrics(model, x_train_ft, y_train_ft, y_train, x_test, y_test, 
                                                  y_pred_test_base, outlier_list, true_coef, normalized_pred_func)
                        
                        # 保存微调训练集散点数据
                        metrics['x_train_ft_scatter'] = x_train_ft_scatter
                        metrics['y_train_ft_scatter'] = y_train_ft_scatter
                        
                        # 保存结果
                        finetune_results[beta][seed][scenario_name][epoch + 1] = metrics
    
    # 保存微调结果
    # 确保所有必需的指标都被保存，包括基线预测结果
    # 为每个结果添加基线预测数据
    for beta in finetune_results:
        for seed in finetune_results[beta]:
            for scenario_name in finetune_results[beta][seed]:
                baseline_metrics = results_base['metrics'][beta][seed][BASELINE_EPOCH]
                for epoch in finetune_results[beta][seed][scenario_name]:
                    # 添加基线预测数据
                    if 'y_pred_test_base' not in finetune_results[beta][seed][scenario_name][epoch]:
                        finetune_results[beta][seed][scenario_name][epoch]['y_pred_test_base'] = baseline_metrics['y_pred_test_base']
                    if 'pred_coef_base' not in finetune_results[beta][seed][scenario_name][epoch]:
                        finetune_results[beta][seed][scenario_name][epoch]['pred_coef_base'] = baseline_metrics['pred_coef_base']
    
    # 保存微调结果
    finetune_results_file = os.path.join(OUTPUT_DIR, "results_finetune.pkl")
    with open(finetune_results_file, 'wb') as f:
        pickle.dump(finetune_results, f)
    print(f"\n保存微调结果到 {finetune_results_file}")
    
    print("\n异常点微调实验完成！")
    print(f"基线结果文件: {baseline_file}")
    print(f"微调结果文件: {finetune_results_file}")


if __name__ == "__main__":
    main()