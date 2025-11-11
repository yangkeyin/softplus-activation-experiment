import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 配置 ---
# 确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置设备为GPU（如果可用）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {DEVICE}')

# 数据配置
N_POINTS = 200  # 信号长度
KEY_FREQS_K = [20, 40, 60]  # k1, k2, k3 - 关键频率分量k
NOISE_LEVEL = 0.1

# 振幅配置（核心）
AMPS_SCENARIO_1 = [1.5, 1.0, 0.5]  # 低频偏置: k1振幅 > k2振幅 > k3振幅
AMPS_SCENARIO_2 = [0.5, 1.0, 1.5]  # 高频偏置: k1振幅 < k2振幅 < k3振幅

# 实验配置
KERNEL_SIZES_TO_TEST = [3, 25, 35]  # 对比 "高通" vs "低通" 两种极端情况
EPOCHS = 2000
EVAL_STEP = 50  # 每50个epoch评估一次相对误差
LR = 0.001
N_SAMPLES_TRAIN = 2000
N_SAMPLES_TEST = 400
SCENARIOS = {"Scenario_1_LowFreqBias": AMPS_SCENARIO_1, "Scenario_2_HighFreqBias": AMPS_SCENARIO_2}

# 输出目录
OUTPUT_DIR = './figures/CNN_freq_bias_denoise_3kernel_noiselevel0.1'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 模型定义 ---
class Simple1DCNN(nn.Module):
    def __init__(self, kernel_size):
        super(Simple1DCNN, self).__init__()
        # 确保卷积后的长度不变
        padding = (kernel_size - 1) // 2
        
        self.conv_stack = nn.Sequential(
            # 输入: (Batch, 1, Length)
            nn.Conv1d(in_channels=1, out_channels=16, 
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, 
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        
        # 使用一个1x1卷积（等效于全连接）来聚合通道
        self.output_layer = nn.Conv1d(in_channels=32, out_channels=1, 
                                      kernel_size=1)
        # 输出: (Batch, 1, Length)

    def forward(self, x):
        # x 形状: (Batch, 1, Length)
        x = self.conv_stack(x)
        x = self.output_layer(x)
        return x


# --- 数据生成函数 ---
def generate_data_scenario(amps_list, key_freqs_k_list, n_samples):
    """
    生成指定场景的数据
    
    参数:
    amps_list: 振幅列表
    key_freqs_k_list: 关键频率分量k列表
    n_samples: 生成的样本数量
    
    返回:
    (X_data, Y_data): 输入和目标数据，形状为 [N, 1, N_POINTS]
    """
    X = []
    Y = []
    
    # 生成时间序列索引
    t = torch.arange(N_POINTS)
    
    # 循环生成样本
    for _ in range(n_samples):
        # 生成干净信号
        y_signal = torch.zeros(N_POINTS)
        for amp, k_freq in zip(amps_list, key_freqs_k_list):
            phase = np.random.uniform(0, 2 * np.pi)
            y_signal += amp * torch.sin(2 * np.pi * k_freq * t / N_POINTS + phase)
        
        # 添加噪声
        noise = torch.randn(N_POINTS) * NOISE_LEVEL
        
        # 保存数据（添加批次和通道维度）
        Y.append(y_signal.reshape(1, N_POINTS))
        X.append((y_signal + noise).reshape(1, N_POINTS))
    
    # 转换为张量
    X_data = torch.stack(X)
    Y_data = torch.stack(Y)
    
    return X_data, Y_data


# --- 辅助函数 ---
def get_avg_spectrum(data_tensor):
    """
    计算数据张量的平均频谱（使用FFT）
    
    参数:
    data_tensor: 形状为 [N, 1, N_POINTS] 的数据张量
    
    返回:
    avg_spectrum: 平均频谱振幅
    """
    # 转换为numpy数组并展平
    data_np = data_tensor.cpu().numpy().squeeze()  # 形状 [N_samples, N_POINTS]
    
    # 计算FFT
    fft_data = np.fft.fft(data_np, axis=1)
    fft_mag = np.abs(fft_data)
    
    # 计算平均频谱并只返回正频率部分
    avg_fft_mag = np.mean(fft_mag, axis=0)
    return avg_fft_mag[:N_POINTS // 2]

def get_avg_relative_error(pred_tensor, target_tensor, key_indices_k):
    """
    计算在关键频率上的平均相对误差
    
    参数:
    pred_tensor: 预测数据，形状为 [N, 1, N_POINTS]
    target_tensor: 目标数据，形状为 [N, 1, N_POINTS]
    key_indices_k: 关键频率分量k的列表
    
    返回:
    avg_errors: 每个关键频率的平均相对误差
    """
    # 计算FFT振幅
    pred_fft_mag = np.abs(np.fft.fft(pred_tensor.cpu().numpy().squeeze(), axis=1))
    target_fft_mag = np.abs(np.fft.fft(target_tensor.cpu().numpy().squeeze(), axis=1))
    
    errors = []
    
    # 计算每个关键频率的相对误差
    for k in key_indices_k:
        true_mag_k = np.mean(target_fft_mag[:, k])  # 在批次上取平均
        pred_mag_k = np.mean(pred_fft_mag[:, k])   # 在批次上取平均
        
        if true_mag_k < 1e-6:
            errors.append(0.0)
        else:
            errors.append(np.abs(pred_mag_k - true_mag_k) / true_mag_k)
    
    return np.array(errors)


# --- 主实验函数 ---
def main():
    # 初始化结果字典
    results = {}
    
    # 关键索引现在就是k值
    key_indices_k = KEY_FREQS_K
    print(f"关键频率分量k: {key_indices_k}")
    
    # 外层循环（遍历场景）
    for scenario_name, amps in SCENARIOS.items():
        print(f"--- 运行场景: {scenario_name} ---")
        results[scenario_name] = {}
        
        # 生成数据
        print(f"  生成训练数据 ({N_SAMPLES_TRAIN} 样本)...")
        X_train, Y_train = generate_data_scenario(amps, KEY_FREQS_K, N_SAMPLES_TRAIN)
        print(f"  生成测试数据 ({N_SAMPLES_TEST} 样本)...")
        X_test, Y_test = generate_data_scenario(amps, KEY_FREQS_K, N_SAMPLES_TEST)
        
        # 将数据移到设备上
        X_train, Y_train = X_train.to(DEVICE), Y_train.to(DEVICE)
        X_test, Y_test = X_test.to(DEVICE), Y_test.to(DEVICE)
        
        # 计算参考频谱
        print("  计算参考频谱...")
        avg_target_spectrum = get_avg_spectrum(Y_test)
        avg_input_spectrum = get_avg_spectrum(X_test)
        results[scenario_name]['avg_target_spectrum'] = avg_target_spectrum
        results[scenario_name]['avg_input_spectrum'] = avg_input_spectrum
        
        # 内层循环（遍历Kernel）
        for kernel_size in KERNEL_SIZES_TO_TEST:
            print(f"--- 正在测试 Kernel Size = {kernel_size} ---")
            
            # 初始化模型
            model = Simple1DCNN(kernel_size=kernel_size).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            criterion = nn.MSELoss()
            
            error_history = []
            
            # 训练模型
            print(f"  开始训练 (EPOCHS={EPOCHS})...")
            for epoch in range(EPOCHS):
                model.train()
                
                # 前向传播
                Y_pred = model(X_train)
                loss = criterion(Y_pred, Y_train)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 周期性评估
                if (epoch + 1) % EVAL_STEP == 0:
                    model.eval()
                    with torch.no_grad():
                        Y_pred_test = model(X_test)
                        avg_errors = get_avg_relative_error(Y_pred_test, Y_test, key_indices_k)
                        error_history.append(avg_errors)
                    
                    if (epoch + 1) % 200 == 0:
                        print(f"    Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}, "
                              f"关键频率误差: {avg_errors}")
            
            # 保存最终结果
            model.eval()
            with torch.no_grad():
                Y_pred_final = model(X_test)
                test_mse = criterion(Y_pred_final, Y_test).item()
                avg_pred_spectrum_final = get_avg_spectrum(Y_pred_final)
            
            results[scenario_name][kernel_size] = {
                'test_mse': test_mse,
                'avg_pred_spectrum': avg_pred_spectrum_final,
                'error_history': np.array(error_history)
            }
            
            print(f"  Kernel Size={kernel_size} 完成! 测试MSE: {test_mse:.6f}")
    
    # 可视化结果
    visualize_results(results)
    
    print("实验完成!")
    return results


# --- 可视化函数 ---
def visualize_results(results):
    """
    可视化实验结果
    """
    for kernel_size in KERNEL_SIZES_TO_TEST:
        # 创建图表
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 2)
        
        # 创建k轴
        k_axis = np.arange(N_POINTS // 2)
        max_k = max(KEY_FREQS_K) * 1.5
        
        # 子图1: 场景1频谱
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(k_axis, results['Scenario_1_LowFreqBias']['avg_input_spectrum'], 'b--', label='Input (Lookback)')
        ax1.plot(k_axis, results['Scenario_1_LowFreqBias']['avg_target_spectrum'], 'g-', label='Ground Truth')
        ax1.plot(k_axis, results['Scenario_1_LowFreqBias'][kernel_size]['avg_pred_spectrum'], 'r-', 
                label=f'Forecasting (k={kernel_size})')
        ax1.set_title('场景 1 (低频偏置) 频谱')
        ax1.set_xlabel('F (Frequency Component k)')
        ax1.set_ylabel('Amplitude')
        ax1.set_xlim(0, max_k)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 场景1误差演化
        ax2 = fig.add_subplot(gs[1, 0])
        data1 = results['Scenario_1_LowFreqBias'][kernel_size]['error_history'].T
        im1 = ax2.imshow(data1, aspect='auto', cmap='gray_r', vmin=np.min(data1), vmax=np.max(data1))
        ax2.set_title('场景 1 相对误差')
        ax2.set_xlabel('评估步骤 (Epoch Step)')
        ax2.set_ylabel('关键频率 (k1, k2, k3)')
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels([f'k={KEY_FREQS_K[0]}', f'k={KEY_FREQS_K[1]}', f'k={KEY_FREQS_K[2]}'])
        
        # 子图3: 场景2频谱
        ax3 = fig.add_subplot(gs[0, 1])
        ax3.plot(k_axis, results['Scenario_2_HighFreqBias']['avg_input_spectrum'], 'b--', label='Input (Lookback)')
        ax3.plot(k_axis, results['Scenario_2_HighFreqBias']['avg_target_spectrum'], 'g-', label='Ground Truth')
        ax3.plot(k_axis, results['Scenario_2_HighFreqBias'][kernel_size]['avg_pred_spectrum'], 'r-', 
                label=f'Forecasting (k={kernel_size})')
        ax3.set_title('场景 2 (高频偏置) 频谱')
        ax3.set_xlabel('F (Frequency Component k)')
        ax3.set_ylabel('Amplitude')
        ax3.set_xlim(0, max_k)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 子图4: 场景2误差演化
        ax4 = fig.add_subplot(gs[1, 1])
        data2 = results['Scenario_2_HighFreqBias'][kernel_size]['error_history'].T
        im2 = ax4.imshow(data2, aspect='auto', cmap='gray_r', vmin=np.min(data2), vmax=np.max(data2))
        ax4.set_title('场景 2 相对误差')
        ax4.set_xlabel('评估步骤 (Epoch Step)')
        ax4.set_ylabel('关键频率 (k1, k2, k3)')
        ax4.set_yticks([0, 1, 2])
        ax4.set_yticklabels([f'k={KEY_FREQS_K[0]}', f'k={KEY_FREQS_K[1]}', f'k={KEY_FREQS_K[2]}'])
        
        # 添加共享的颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im1, cax=cbar_ax, label='相对误差')
        
        # 设置主标题
        plt.suptitle(f'CNN 偏见分析 - Kernel Size = {kernel_size}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.92, 0.96])
        
        # 保存图表
        filename = os.path.join(OUTPUT_DIR, f'cnn_bias_k{kernel_size}_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"保存图表到: {filename}")
        
        # 显示图表
        plt.close()


if __name__ == "__main__":
    main()