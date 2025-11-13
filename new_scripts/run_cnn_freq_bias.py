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
plt.rcParams['font.sans-serif'] =  ['Microsoft YaHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置设备为GPU（如果可用）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {DEVICE}')

# 数据配置 - 预测长度 = 历史长度 (L=H)
SEQ_LEN = 250  # 历史序列长度 L
PRED_LEN = 250  # 预测序列长度 H, H = L
TOTAL_SERIES_LENGTH = 10000  # 用于生成窗口的原始序列总长
N_POINTS = SEQ_LEN  # 为了兼容原有代码
KEY_FREQS_K = [20, 40, 60]  # k1, k2, k3 - 关键频率分量k
NOISE_LEVEL = 0.5

# 振幅配置（核心）
AMPS_SCENARIO_1 = [1.5, 1.0, 0.5]  # 低频偏置: k1振幅 > k2振幅 > k3振幅
AMPS_SCENARIO_2 = [0.5, 1.0, 1.5]  # 高频偏置: k1振幅 < k2振幅 < k3振幅

# 实验配置
KERNEL_SIZES_TO_TEST = [3, 25, 35]  # 对比 "高通" vs "低通" 两种极端情况
EPOCHS = 2000
EVAL_STEP = 1  # 每50个epoch评估一次相对误差
LR = 0.001
BATCH_SIZE = 64  # 批量大小
N_SAMPLES_TRAIN = 2000
N_SAMPLES_TEST = 400
SCENARIOS = {"Scenario_1_LowFreqBias": AMPS_SCENARIO_1, "Scenario_2_HighFreqBias": AMPS_SCENARIO_2}

# 可视化配置
NUM_XTICKS = 10  # x轴显示的刻度数量

# 输出目录
OUTPUT_DIR = './figures/CNN_freq_bias_denoise_amp0.5-1.5_epo2000_noiselevel0.5'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 模型定义 ---
class Simple1DCNN(nn.Module):
    def __init__(self, kernel_size, seq_len, pred_len):
        super(Simple1DCNN, self).__init__()
        # 确保卷积后的长度不变
        padding = (kernel_size - 1) // 2
        
        self.conv_stack = nn.Sequential(
            # 输入: (Batch, 1, SEQ_LEN)
            nn.Conv1d(in_channels=1, out_channels=16, 
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, 
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU()
            # 输出: (Batch, 32, SEQ_LEN)
        )
        
        # 输出层：将 (32, SEQ_LEN) 的特征映射到 (1, PRED_LEN) 的预测
        self.output_layer = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1), # 聚合通道
            # 输出: (Batch, 1, SEQ_LEN)
            nn.Flatten(),
            # 输出: (Batch, SEQ_LEN)
            nn.Linear(seq_len, pred_len) # 全连接层 (seq_len == pred_len)
            # 输出: (Batch, PRED_LEN)
        )

    def forward(self, x):
        # x 形状: (Batch, 1, SEQ_LEN)
        x = self.conv_stack(x)
        x = self.output_layer(x)
        # 添加通道维度，最终输出: (Batch, 1, PRED_LEN)
        x = x.unsqueeze(1)
        return x


# --- 数据生成函数 ---
def generate_time_series(amps_list, key_freqs_k_list, total_length):
    """
    先生成长序列
    
    参数:
    amps_list: 振幅列表
    key_freqs_k_list: 关键频率分量k列表
    total_length: 生成的时间序列总长度
    
    返回:
    full_series: 生成的完整时间序列
    """
    t = torch.arange(total_length)
    series = torch.zeros(total_length)
    
    # 生成干净信号
    for amp, k_freq in zip(amps_list, key_freqs_k_list):
        # 频率 f 通过 k / SEQ_LEN 计算
        freq = k_freq / SEQ_LEN
        series += amp * torch.sin(2 * np.pi * freq * t)
    
    # 添加噪声
    noise = torch.randn(total_length) * NOISE_LEVEL
    noisy_series = series + noise
    
    return noisy_series, series

def create_sliding_windows(series, lookback, forecast):
    """
    创建滑动窗口数据
    
    参数:
    series: 输入时间序列
    lookback: 历史窗口长度
    forecast: 预测窗口长度
    
    返回:
    X, Y: 输入和目标数据
    """
    X, Y = [], []
    # 确保有足够的数据创建窗口
    max_start_idx = len(series) - lookback - forecast + 1
    
    for i in range(max_start_idx):
        # 确保获取的是numpy数组或Python列表形式的数据
        window_x = series[i:i+lookback]
        window_y = series[i+lookback:i+lookback+forecast]
        
        # 如果是torch张量，转换为numpy数组
        if isinstance(window_x, torch.Tensor):
            window_x = window_x.cpu().numpy()
        if isinstance(window_y, torch.Tensor):
            window_y = window_y.cpu().numpy()
        
        X.append(window_x)
        Y.append(window_y)
    
    # 转换为numpy数组然后再转为张量，避免类型转换错误
    X_array = np.array(X)
    Y_array = np.array(Y)
    
    # 转换为张量并添加通道维度
    X_tensor = torch.tensor(X_array).unsqueeze(1)
    Y_tensor = torch.tensor(Y_array).unsqueeze(1)
    
    return X_tensor, Y_tensor

def create_datasets(amps_list, key_freqs_k_list, train_ratio=0.7, val_ratio=0.1):
    """
    创建训练、验证和测试数据集
    
    参数:
    amps_list: 振幅列表
    key_freqs_k_list: 关键频率分量k列表
    train_ratio: 训练集比例
    val_ratio: 验证集比例
    
    返回:
    train_loader, val_loader, test_loader, X_test, Y_test: 数据加载器和测试数据
    """
    # 生成完整的时间序列
    noisy_series, clean_series = generate_time_series(amps_list, key_freqs_k_list, TOTAL_SERIES_LENGTH)
    
    # 分割数据
    train_size = int(len(noisy_series) * train_ratio)
    val_size = int(len(noisy_series) * val_ratio)
    
    train_data = noisy_series[:train_size]
    val_data = noisy_series[train_size:train_size+val_size]
    test_data = noisy_series[train_size+val_size:]
    
    # 创建滑动窗口
    X_train, Y_train = create_sliding_windows(train_data, SEQ_LEN, PRED_LEN)
    X_val, Y_val = create_sliding_windows(val_data, SEQ_LEN, PRED_LEN)
    X_test, Y_test = create_sliding_windows(test_data, SEQ_LEN, PRED_LEN)
    
    # 限制样本数量，防止数据过多
    if len(X_train) > N_SAMPLES_TRAIN:
        X_train = X_train[:N_SAMPLES_TRAIN]
        Y_train = Y_train[:N_SAMPLES_TRAIN]
    
    if len(X_test) > N_SAMPLES_TEST:
        X_test = X_test[:N_SAMPLES_TEST]
        Y_test = Y_test[:N_SAMPLES_TEST]
    
    # 创建DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, X_test, Y_test


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
    ax[1].plot(avg_fft_mag[:N_POINTS // 2])
    ax[1].set_title('Average FFT Spectrum')
    ax[1].set_ylabel('Amplitude')
    fig.tight_layout()
    fig.show()
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
        print("  创建数据集...")
        train_loader, val_loader, X_test, Y_test = create_datasets(amps, KEY_FREQS_K)
        
        # 将测试数据移到设备上
        X_test, Y_test = X_test.to(DEVICE), Y_test.to(DEVICE)
        
        # 计算参考频谱
        print("  计算参考频谱...")
        avg_target_spectrum = get_avg_spectrum(Y_test)
        avg_input_spectrum = get_avg_spectrum(X_test)
        results[scenario_name]['avg_target_spectrum'] = avg_target_spectrum
        results[scenario_name]['avg_input_spectrum'] = avg_input_spectrum
        # 保存序列长度信息
        results[scenario_name]['seq_len'] = SEQ_LEN
        results[scenario_name]['pred_len'] = PRED_LEN
        
        # 内层循环（遍历Kernel）
        for kernel_size in KERNEL_SIZES_TO_TEST:
            print(f"--- 正在测试 Kernel Size = {kernel_size} ---")
            
            # 初始化模型
            model = Simple1DCNN(kernel_size=kernel_size, seq_len=SEQ_LEN, pred_len=PRED_LEN).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            criterion = nn.MSELoss()
            
            error_history = []
            
            # 训练模型
            print(f"  开始训练 (EPOCHS={EPOCHS})...")
            for epoch in range(EPOCHS):
                model.train()
                epoch_loss = 0
                
                for batch_X, batch_Y in train_loader:
                    batch_X, batch_Y = batch_X.to(DEVICE), batch_Y.to(DEVICE)
                    
                    # 前向传播
                    Y_pred = model(batch_X)
                    loss = criterion(Y_pred, batch_Y)
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item() * batch_X.size(0)
                
                # 计算平均损失
                epoch_loss /= len(train_loader.dataset)
                
                # 周期性评估
                if (epoch + 1) % EVAL_STEP == 0:
                    model.eval()
                    with torch.no_grad():
                        Y_pred_test = model(X_test)
                        avg_errors = get_avg_relative_error(Y_pred_test, Y_test, key_indices_k)
                        error_history.append(avg_errors)
                        print(f"    Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.6f}, "
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
        im1 = ax2.imshow(data1, aspect='auto', cmap='gray_r', vmin=0, vmax=0.95)
        ax2.set_title('场景 1 相对误差')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('关键频率 (k1, k2, k3)')
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels([f'k={KEY_FREQS_K[0]}', f'k={KEY_FREQS_K[1]}', f'k={KEY_FREQS_K[2]}'])
        # 精简x轴标签，避免过多
        epoch_labels = np.arange(EVAL_STEP, EPOCHS+1, EVAL_STEP)
        step = max(1, len(epoch_labels) // NUM_XTICKS)
        ax2.set_xticks(np.arange(0, len(epoch_labels), step))
        ax2.set_xticklabels(epoch_labels[::step])
        
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
        im2 = ax4.imshow(data2, aspect='auto', cmap='gray_r', vmin=0, vmax=0.95)
        ax4.set_title('场景 2 相对误差')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('关键频率 (k1, k2, k3)')
        ax4.set_yticks([0, 1, 2])
        ax4.set_yticklabels([f'k={KEY_FREQS_K[0]}', f'k={KEY_FREQS_K[1]}', f'k={KEY_FREQS_K[2]}'])
        # 精简x轴标签，避免过多
        epoch_labels = np.arange(EVAL_STEP, EPOCHS+1, EVAL_STEP)
        step = max(1, len(epoch_labels) // NUM_XTICKS)
        ax4.set_xticks(np.arange(0, len(epoch_labels), step))
        ax4.set_xticklabels(epoch_labels[::step])
        
        # 添加共享的颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im1, cax=cbar_ax, label='相对误差')
        
        # 设置主标题
        plt.suptitle(f'CNN 偏见分析 - Kernel Size = {kernel_size} - L=H=200', fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.92, 0.96])
        
        # 保存图表
        filename = os.path.join(OUTPUT_DIR, f'cnn_bias_k{kernel_size}_prediction_L200_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"保存图表到: {filename}")
        
        # 显示图表
        plt.close()


if __name__ == "__main__":
    main()
