import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os

# --- 0. 配置 ---
# 确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

# 设置设备为GPU（如果可用）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {DEVICE}')

# 定义序列参数
SEQ_LEN = 250  # 输入序列长度
PRED_LEN = 250   # 预测序列长度
TOTAL_LEN = SEQ_LEN + PRED_LEN

# 训练参数
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.0001

OUTPUT_DIR = './figures/Transformer_amplitude'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 1. 合成数据生成 ---

def generate_time_series(length, amplitudes, frequencies, noise_level=0.5):
    """
    生成一个合成时间序列，作为多个正弦波和噪声的叠加。
    
    参数:
    length (int): 序列总长度
    amplitudes (list): 每个频率分量的振幅
    frequencies (list): 每个频率分量的频率 (单位: 1/序列长度)
    noise_level (float): 高斯噪声的标准差
    
    返回:
    numpy.ndarray: 生成的时间序列
    """
    t = np.arange(length)
    series = np.zeros(length)
    for amp, freq in zip(amplitudes, frequencies):
        series += amp * np.sin(2 * np.pi * freq * t / length)
    series += np.random.normal(0, noise_level, length)
    return series

def get_key_freq_indices(frequencies, length):
    """根据频率计算它们在FFT结果中的索引"""
    # 频率 f 对应的索引 k 约为 k = f * L / f_s
    # 在我们的例子中, f_s = L (因为我们用 1/L 为单位), 所以 k 约等于 f
    # 为了稳健性，我们将在FFT频谱中搜索最接近这些频率的峰值
    # 为简单起见，我们直接使用频率值作为索引的近似值
    # 注意：这是一个简化处理，在实际应用中需要更精确的峰值匹配
    return [int(f) for f in frequencies]

def create_sliding_windows(data, seq_len, pred_len):
    """从时间序列创建输入(X)和目标(Y)的滑动窗口"""
    x, y = [], []
    total_len = len(data) - seq_len - pred_len + 1
    for i in range(total_len):
        x.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + pred_len])
    return np.array(x), np.array(y)

# --- 2. 简化的 Transformer 模型 ---

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SimpleTransformerModel(nn.Module):
    """一个简化的基于Transformer Encoder的预测模型"""
    def __init__(self, d_model=32, nhead=4, num_layers=2, input_dim=1, output_dim=1, seq_len=SEQ_LEN, pred_len=PRED_LEN):
        super(SimpleTransformerModel, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 输入嵌入层 (将单变量时间序列映射到d_model维度)
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer 编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 输出层 (将d_model维度映射回预测长度)
        # 我们将 (Batch, SeqLen, d_model) -> (Batch, SeqLen * d_model) -> (Batch, PredLen)
        self.flatten = nn.Flatten()
        self.output_layer = nn.Linear(seq_len * d_model, pred_len * output_dim)
        self.output_dim = output_dim
        
        # 将模型移至设备
        self.to(DEVICE)

    def forward(self, x):
        # x 形状: (Batch, SeqLen, InputDim)
        x = self.input_embedding(x)  # (Batch, SeqLen, d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1) # (Batch, SeqLen, d_model)
        x = self.transformer_encoder(x) # (Batch, SeqLen, d_model)
        x = self.flatten(x) # (Batch, SeqLen * d_model)
        x = self.output_layer(x) # (Batch, PredLen * OutputDim)
        # 重塑为 (Batch, PredLen, OutputDim)
        return x.view(-1, self.pred_len, self.output_dim)

# --- 3. 训练与分析辅助函数 ---

def get_relative_error(pred_fft, true_fft, key_indices):
    """
    计算关键频率分量上的相对误差 (Delta_k)
    Delta_k = |a'_k - a_k| / |a_k|
     [cite: 164]
    """
    errors = []
    for k in key_indices:
        # 确保 |a_k| 不为零，避免除零错误
        true_mag = np.abs(true_fft[k])
        if true_mag < 1e-6:
            errors.append(0.0) # 如果真实幅度接近0，则误差为0
        else:
            error = np.abs(pred_fft[k] - true_fft[k]) / true_mag
            errors.append(error)
    return np.array(errors)

def normalize_fft(fft_data, norm_type='none'):
    """在频域中对FFT系数进行归一化"""
    if norm_type == 'none':
        return fft_data
    
    if norm_type == 'zscore_mag':
        # 对振幅谱进行Z-score归一化
         # [cite: 191] "normalize the amplitudes of the frequencies to eliminate their proportional differences"
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)
        
        mean = np.mean(magnitude, axis=-1, keepdims=True)
        std = np.std(magnitude, axis=-1, keepdims=True)
        std[std < 1e-6] = 1.0 # 避免除零
        
        norm_magnitude = (magnitude - mean) / std
        
        # 重新组合复数
        return norm_magnitude * np.exp(1j * phase)

    raise ValueError(f"未知的归一化类型: {norm_type}")

def train_and_analyze(
    model, 
    data, 
    key_frequencies, 
    preprocess_mode='time_non_norm'
):
    """
    训练模型并跟踪关键频率的相对误差。
    
    preprocess_mode (str):
    'time_non_norm':      时域模型 + 非归一化 (Fig 2b 左)
    'time_norm':          时域模型 + 归一化 (Fig 2b 中)
    'freq_norm':          频域模型 + 归一化 (Fig 2b 右)
    """
    
    # --- 1. 数据准备 ---

    # <--- 修改：按时间顺序分割数据集 ---
    print("正在按 70/10/20 比例分割数据集...")
    total_data_len = len(data)
    train_split_idx = int(total_data_len * 0.7)
    val_split_idx = int(total_data_len * 0.8)

    # 确保分割点不会切断一个完整的样本
    # (这一步对于纯滑动窗口是可选的，但对于真实数据是好习惯)

    data_train = data[:train_split_idx]
    data_val = data[train_split_idx:val_split_idx]
    data_test = data[val_split_idx:]

    print(f"训练集大小: {len(data_train)}, 验证集大小: {len(data_val)}, 测试集大小: {len(data_test)}")

    # <--- 修改：在 *各自* 的集合内创建窗口 ---
    x_train_data, y_train_data = create_sliding_windows(data_train, SEQ_LEN, PRED_LEN)
    x_val_data, y_val_data = create_sliding_windows(data_val, SEQ_LEN, PRED_LEN)
    x_test_data, y_test_data = create_sliding_windows(data_test, SEQ_LEN, PRED_LEN)

    # <--- 修改：测试样本现在来自 *测试集* ---
    x_test_sample = x_test_data[-1]
    y_test_sample = y_test_data[-1]

    # (注意：如果 x_test_data 为空（因为测试集太短），这里会报错，但对于 2000 个点是足够的)
    
    # 根据模式对 *训练* 数据进行预处理
    if preprocess_mode == 'time_non_norm':
         # [cite: 189] (Fig 2b 左)
        x_train_tensor = torch.tensor(x_train_data, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        y_train_tensor = torch.tensor(y_train_data, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        
    elif preprocess_mode == 'time_norm':
         # [cite: 192] (Fig 2b 中)
        # DFT -> Norm -> IDFT -> 输入模型
        x_fft = np.fft.fft(x_train_data, axis=1)
        x_fft_norm = normalize_fft(x_fft, 'zscore_mag')
        x_data_norm = np.fft.ifft(x_fft_norm, axis=1).real
        
        # 目标Y保持不变
        x_train_tensor = torch.tensor(x_data_norm, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        y_train_tensor = torch.tensor(y_train_data, dtype=torch.float32).unsqueeze(-1).to(DEVICE)

    elif preprocess_mode == 'freq_norm':
         # [cite: 195, 197] (Fig 2b 右)
        # DFT -> Norm -> 输入模型 (模型在频域工作)
        x_fft = np.fft.fft(x_train_data, axis=1)
        x_fft_norm = normalize_fft(x_fft, 'zscore_mag')
        
        y_fft = np.fft.fft(y_train_data, axis=1)
        y_fft_norm = normalize_fft(y_fft, 'zscore_mag')
        
        # 我们需要将复数 (a+bi) 转换为2个通道 (a, b)
        x_data_freq = np.stack([x_fft_norm.real, x_fft_norm.imag], axis=-1)
        y_data_freq = np.stack([y_fft_norm.real, y_fft_norm.imag], axis=-1)

        x_train_tensor = torch.tensor(x_data_freq, dtype=torch.float32).to(DEVICE)
        y_train_tensor = torch.tensor(y_data_freq, dtype=torch.float32).to(DEVICE)

    else:
        raise ValueError("未知的预处理模式")

    # <--- 修改：DataLoader 只使用训练集 ---
    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # 添加验证集预处理和验证损失计算
    # 为验证集准备数据
    if preprocess_mode == 'time_non_norm':
        x_val_tensor = torch.tensor(x_val_data, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        y_val_tensor = torch.tensor(y_val_data, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    elif preprocess_mode == 'time_norm':
        x_val_fft = np.fft.fft(x_val_data, axis=1)
        x_val_fft_norm = normalize_fft(x_val_fft, 'zscore_mag')
        x_val_data_norm = np.fft.ifft(x_val_fft_norm, axis=1).real
        x_val_tensor = torch.tensor(x_val_data_norm, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        y_val_tensor = torch.tensor(y_val_data, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    elif preprocess_mode == 'freq_norm':
        x_val_fft = np.fft.fft(x_val_data, axis=1)
        x_val_fft_norm = normalize_fft(x_val_fft, 'zscore_mag')
        y_val_fft = np.fft.fft(y_val_data, axis=1)
        y_val_fft_norm = normalize_fft(y_val_fft, 'zscore_mag')
        x_val_freq = np.stack([x_val_fft_norm.real, x_val_fft_norm.imag], axis=-1)
        y_val_freq = np.stack([y_val_fft_norm.real, y_val_fft_norm.imag], axis=-1)
        x_val_tensor = torch.tensor(x_val_freq, dtype=torch.float32).to(DEVICE)
        y_val_tensor = torch.tensor(y_val_freq, dtype=torch.float32).to(DEVICE)
    
    key_indices_input = get_key_freq_indices(key_frequencies, SEQ_LEN)
    key_indices_pred = get_key_freq_indices(key_frequencies, PRED_LEN)
    
    error_history = [] # 存储 (EPOCHS, num_key_freqs) 的误差
    
    # --- 2. 训练循环 ---
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pred = model(x_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()
        
        train_loss_avg = train_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, 训练损失: {train_loss_avg:.4f}, 验证损失: {val_loss:.4f}")
        
        # --- 3. 周期性分析 (计算相对误差) ---
        model.eval()
        true_y_time = None
        with torch.no_grad():
            # 准备分析用的测试样本 (x_test_sample, y_test_sample)
            if preprocess_mode == 'time_non_norm':
                x_test_tensor = torch.tensor(x_test_sample, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
                pred_y_time = model(x_test_tensor).squeeze().cpu().numpy()  # 移回CPU进行numpy操作
                true_y_time = y_test_sample
            
            elif preprocess_mode == 'time_norm':
                x_test_fft = np.fft.fft(x_test_sample)
                x_test_fft_norm = normalize_fft(x_test_fft, 'zscore_mag')
                x_test_data_norm = np.fft.ifft(x_test_fft_norm).real
                x_test_tensor = torch.tensor(x_test_data_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
                
                pred_y_time = model(x_test_tensor).squeeze().cpu().numpy()  # 移回CPU进行numpy操作
                true_y_time = y_test_sample
                
            elif preprocess_mode == 'freq_norm':
                x_test_fft = np.fft.fft(x_test_sample)
                x_test_fft_norm = normalize_fft(x_test_fft, 'zscore_mag')
                x_test_freq = np.stack([x_test_fft_norm.real, x_test_fft_norm.imag], axis=-1)
                x_test_tensor = torch.tensor(x_test_freq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                pred_y_freq = model(x_test_tensor).squeeze().cpu().numpy()  # 移回CPU进行numpy操作
                pred_y_fft_norm = pred_y_freq[..., 0] + 1j * pred_y_freq[..., 1]
                
                # 我们需要 "去归一化" 才能与真实值比较，但论文中似乎是直接比较
                # 为简单起见，我们直接在时域比较
                #  IDFT -> 时域预测 [cite: 197]
                pred_y_time = np.fft.ifft(pred_y_fft_norm).real # 假设去归一化步骤被省略
                true_y_time = y_test_sample
            
            # 计算预测和真实值的FFT
            pred_y_fft = np.fft.fft(pred_y_time) # 预测值的FFT
            true_y_fft = np.fft.fft(true_y_time) # 真实值的FFT
            
            # 计算相对误差
            errors = get_relative_error(pred_y_fft, true_y_fft, key_indices_pred)
            
            # <--- DEBUG 1: 检查 Loop 3 ---
            # 只在最后一个 epoch，且是第一个测试样本时打印
            if epoch == EPOCHS - 1:
                print(f"\n--- DEBUG (LOOP 3 / Epoch {epoch}) ---")
                diff_sum = np.sum(np.abs(true_y_time - pred_y_time))
                print(f"Loop 3 (Heatmap) Diff Sum: {diff_sum}")
                if diff_sum == 0.0:
                    print(">>> Loop 3 发现 0 误差!")
                else:
                    print(">>> Loop 3 发现 *非零* 误差 (正常)")
            
            error_history.append(errors)

    # --- 4. 准备最终绘图数据 ---
    error_history = np.array(error_history)
    
    # 准备最终的绘图数据 (Input, Ground Truth, Forecasting)
    # 我们将使用 (x_test_sample + y_test_sample) 作为 Ground Truth
    ground_truth_time = np.concatenate([x_test_sample, y_test_sample])
    input_time = np.concatenate([x_test_sample, np.full(PRED_LEN, np.nan)])
    
    # 获取最终预测
    print("训练结束，开始预测")
    model.eval()
    with torch.no_grad():
        if preprocess_mode == 'time_non_norm':
            x_test_tensor = torch.tensor(x_test_sample, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            pred_y_time = model(x_test_tensor).squeeze().cpu().numpy()  # 移回CPU进行numpy操作
        
        elif preprocess_mode == 'time_norm':
            x_test_fft = np.fft.fft(x_test_sample)
            x_test_fft_norm = normalize_fft(x_test_fft, 'zscore_mag')
            x_test_data_norm = np.fft.ifft(x_test_fft_norm).real
            x_test_tensor = torch.tensor(x_test_data_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            pred_y_time = model(x_test_tensor).squeeze().cpu().numpy()  # 移回CPU进行numpy操作  

        elif preprocess_mode == 'freq_norm':
            x_test_fft = np.fft.fft(x_test_sample)
            x_test_fft_norm = normalize_fft(x_test_fft, 'zscore_mag')
            x_test_freq = np.stack([x_test_fft_norm.real, x_test_fft_norm.imag], axis=-1)
            x_test_tensor = torch.tensor(x_test_freq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred_y_freq = model(x_test_tensor).squeeze().cpu().numpy()  # 移回CPU进行numpy操作
            pred_y_fft_norm = pred_y_freq[..., 0] + 1j * pred_y_freq[..., 1]
            pred_y_time = np.fft.ifft(pred_y_fft_norm).real

    forecast_time = np.concatenate([np.full(SEQ_LEN, np.nan), pred_y_time])

    # 计算FFT用于绘图 (在整个窗口上)
     # [cite: 113] "We use DFT to analyze the frequency content of X, X_hat and X'"
    # X_hat 是 Ground Truth, X' 是 Forecasting
    
    # <--- 修改：应用 "先填充，后加窗" 方案进行绘图 ---
    print("应用 '先填充，后加窗' 方案准备绘图数据...")

    # 1. 定义 *统一的* N=SEQ_LEN 窗函数
    win_common = np.hanning(SEQ_LEN)

    # 2. 零填充至统一长度 (SEQ_LEN)
    # Input (x_test_sample) 已经是 SEQ_LEN，无需填充
    true_padded = np.pad(y_test_sample, (0, SEQ_LEN - PRED_LEN), 'constant')
    pred_padded = np.pad(pred_y_time, (0, SEQ_LEN - PRED_LEN), 'constant')

    # 3. 对 *所有* N=SEQ_LEN 的信号应用 *同一个* 窗
    #    (x_test_sample 是 N=192)
    #    (true_padded 是 N=192)
    #    (pred_padded 是 N=192)
    input_windowed = x_test_sample * win_common
    true_windowed = true_padded * win_common
    pred_windowed = pred_padded * win_common
    
    # <--- DEBUG 2: 检查 Loop 4 ---
    # 只在第一个测试样本时打印
    print(f"\n--- DEBUG (LOOP 4 / Plotting) ---")
    diff_sum = np.sum(np.abs(true_y_time - pred_y_time))
    print(f"Loop 4 (FFT Plot) Diff Sum: {diff_sum}")
    if diff_sum == 0.0:
        print(">>> Loop 4 发现 0 误差! (这就是 Bug!)")
    else:
        print(">>> Loop 4 发现 *非零* 误差 (正常)")

    # 4. 计算 FFT 幅度
    input_fft_mag = np.abs(np.fft.fft(input_windowed))
    true_fft_mag = np.abs(np.fft.fft(true_windowed)) 
    pred_fft_mag = np.abs(np.fft.fft(pred_windowed))

    # 5. 定义统一的频率轴 (基于 SEQ_LEN)
    fft_freq_common = np.fft.fftfreq(SEQ_LEN, d=1)

    # 6. 取FFT的前半部分 (N/2)
    N_common = SEQ_LEN // 2

    plot_data = {
        'input_fft_mag': input_fft_mag[:N_common],
        'true_fft_mag': true_fft_mag[:N_common],
        'pred_fft_mag': pred_fft_mag[:N_common],
        'freq_axis_common': fft_freq_common[:N_common] * SEQ_LEN, # 转换为 k
    }

    return error_history, plot_data

# --- 4. 复现 Figure 2(a) ---
# # [cite: 177] "Investigating the Frequency Bias of the Transformer (Case 1)"
# # [cite: 168] "two datasets with three key frequency components ({k1,k2, k3})"

def run_figure_2a():
    print("--- 正在运行 Figure 2(a) ---")
    
    # 定义3个关键频率 - 分析目标（相对于SEQ_LEN=192）
    key_freqs_2a = [30, 70, 120] # k1, k2, k3
    
    # 场景 1 (左图): 振幅 k1 > k2 > k3
     # [cite: 171] 论文中描述 $a_{k1}<a_{k2}<a_{k3}$，但左图显示 k1 振幅最高。
    # 我们复现图表所显示的视觉效果。
    print("运行场景 1 (左图): Amp(k1) > Amp(k2) > Amp(k3)")
    amps_left = [15.0, 10.0, 5.0]
    # 计算用于 N=10000 数据生成的"工厂频率"
    # f = k_lab / N_lab = k_factory / N_factory
    # k_factory = (k_lab / N_lab) * N_factory
    data_gen_freqs_left = [(k / SEQ_LEN) * 10000 for k in key_freqs_2a]
    data_left = generate_time_series(10000, amps_left, data_gen_freqs_left, noise_level=5.0)
     # [cite: 167] "employ a Transformer model [29]"
    model_left = SimpleTransformerModel(
        d_model=16, nhead=4, num_layers=1, 
        input_dim=1, output_dim=1, 
        seq_len=SEQ_LEN, pred_len=PRED_LEN
    )
    errors_left, plot_left = train_and_analyze(
        model_left, data_left, key_freqs_2a, 'time_non_norm'
    )
    
    # 场景 2 (右图): 振幅 k1 < k2 < k3
     # [cite: 185] "synthetic data with higher amplitudes in the mid and high-frequency ranges"
    print("运行场景 2 (右图): Amp(k1) < Amp(k2) < Amp(k3)")
    amps_right = [5.0, 10.0, 15.0]
    # 计算用于 N=10000 数据生成的"工厂频率"
    data_gen_freqs_right = [(k / SEQ_LEN) * 10000 for k in key_freqs_2a]
    data_right = generate_time_series(10000, amps_right, data_gen_freqs_right, noise_level=5.0)
    model_right = SimpleTransformerModel(
        d_model=16, nhead=4, num_layers=1, 
        input_dim=1, output_dim=1, 
        seq_len=SEQ_LEN, pred_len=PRED_LEN
    )
    errors_right, plot_right = train_and_analyze(
        model_right, data_right, key_freqs_2a, 'time_non_norm'
    )

    # --- 绘图 ---
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2)
    
    # --- 左图 (振幅图) ---
    ax_left_fft = fig.add_subplot(gs[0, 0])
    # 使用统一的频率轴绘制所有FFT
    ax_left_fft.plot(plot_left['freq_axis_common'], plot_left['true_fft_mag'], 'g-', label='Ground Truth', linewidth=2)
    ax_left_fft.plot(plot_left['freq_axis_common'], plot_left['pred_fft_mag'], 'r-', label='Forecasting', linewidth=2)
    # 绘制输入用于对比
    ax_left_fft.plot(plot_left['freq_axis_common'], plot_left['input_fft_mag'], 'b--', label='Input (Lookback)', alpha=0.7)
    ax_left_fft.set_title("Fig 2(a) Left: $A(k1) = A(k2) = A(k3)$")
    ax_left_fft.set_xlabel("F (Frequency Component k)")
    ax_left_fft.set_ylabel("Amplitude")
    ax_left_fft.legend()
    # ax_left_fft.set_xlim(2, max(key_freqs_2a) * 1.5)

    # --- 左图 (热力图) ---
    ax_left_heatmap = fig.add_subplot(gs[1, 0])
     # [cite: 160] "relative error Delta_k"
    im = ax_left_heatmap.imshow(errors_left.T, aspect='auto', cmap='gray_r', vmin=errors_left.min(), vmax=errors_left.max())
    ax_left_heatmap.set_yticks(range(len(key_freqs_2a)))
    ax_left_heatmap.set_yticklabels([f"k{i+1}" for i in range(len(key_freqs_2a))])
    ax_left_heatmap.set_xlabel("#epoch")
    ax_left_heatmap.set_ylabel("Relative Error")

    # --- 右图 (振幅图) ---
    ax_right_fft = fig.add_subplot(gs[0, 1])
    ax_right_fft.plot(plot_right['freq_axis_common'], plot_right['true_fft_mag'], 'g-', label='Ground Truth', linewidth=2)
    ax_right_fft.plot(plot_right['freq_axis_common'], plot_right['pred_fft_mag'], 'r-', label='Forecasting', linewidth=2)
    ax_right_fft.plot(plot_right['freq_axis_common'], plot_right['input_fft_mag'], 'b--', label='Input (Lookback)', alpha=0.7)
    ax_right_fft.set_title("Fig 2(a) Right: $A(k1) = A(k2) = A(k3)$")
    ax_right_fft.set_xlabel("F (Frequency Component k)")
    ax_right_fft.set_ylabel("Amplitude")
    ax_right_fft.legend()
    # ax_right_fft.set_xlim(2, max(key_freqs_2a) * 1.5)

    # --- 右图 (热力图) ---
    ax_right_heatmap = fig.add_subplot(gs[1, 1])
    im = ax_right_heatmap.imshow(errors_right.T, aspect='auto', cmap='gray_r', vmin=errors_right.min(), vmax=errors_right.max())
    ax_right_heatmap.set_yticks(range(len(key_freqs_2a)))
    ax_right_heatmap.set_yticklabels([f"k{i+1}" for i in range(len(key_freqs_2a))])
    ax_right_heatmap.set_xlabel("#epoch")
    ax_right_heatmap.set_ylabel("Relative Error")
    
    fig.subplots_adjust(right=0.85)
    fig.colorbar(im, cax=fig.add_axes([0.88, 0.15, 0.04, 0.7]), label="Relative Error (0=black, 1=white)")
    plt.suptitle("Figure 2(a) Conceptual Reproduction: Frequency Bias in Transformers", y=1.03)
    plt.show()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figure_2a_sameamp.png'))

# --- 5. 复现 Figure 2(b) ---
 # [cite: 188] "Debiasing the Frequency Learning for the Transformer. (Case 2)"
 # [cite: 175] "dataset with four key frequency components"

def run_figure_2b():
    print("--- 正在运行 Figure 2(b) ---")
    
    # 4个关键频率 - 分析目标（相对于SEQ_LEN=192）
    key_freqs_2b = [10, 20, 30, 40]
    # 计算用于 N=10000 数据生成的"工厂频率"
    data_gen_freqs = [(k / SEQ_LEN) * 10000 for k in key_freqs_2b]
    # 振幅递减
    amps_2b = [20.0, 15.0, 10.0, 5.0]
    data_2b = generate_time_series(10000, amps_2b, data_gen_freqs, noise_level=2.0)
    
    # 实验 1: 时域模型 + 非归一化
     # [cite: 189] "Time domain model + Non-normalization"
    print("运行实验 1: Time domain model + Non-normalization")
    model_b1 = SimpleTransformerModel(
        d_model=32, nhead=4, num_layers=2, 
        input_dim=1, output_dim=1, 
        seq_len=SEQ_LEN, pred_len=PRED_LEN
    )
    _, plot_b1 = train_and_analyze(
        model_b1, data_2b, key_freqs_2b, 'time_non_norm'
    )
    
    # 实验 2: 时域模型 + 归一化
     # [cite: 191] "Frequency normalization"
     # [cite: 192] "IDFT to convert... back into the time domain before inputting it"
    print("运行实验 2: Time domain model + Normalization")
    model_b2 = SimpleTransformerModel(
        d_model=32, nhead=4, num_layers=2, 
        input_dim=1, output_dim=1, 
        seq_len=SEQ_LEN, pred_len=PRED_LEN
    )
    _, plot_b2 = train_and_analyze(
        model_b2, data_2b, key_freqs_2b, 'time_norm'
    )

    # 实验 3: 频域模型 + 归一化
     # [cite: 195] "directly deploy the Transformer on the frequency domain"
    print("运行实验 3: Frequency domain + Normalization")
    # 模型输入和输出都是 (Batch, SeqLen, 2) (实部/虚部)
    model_b3 = SimpleTransformerModel(
        d_model=32, nhead=4, num_layers=2, 
        input_dim=2, output_dim=2, # 输入输出为 (real, imag)
        seq_len=SEQ_LEN, pred_len=PRED_LEN # 在频域中，长度也是一样的
    )
    _, plot_b3 = train_and_analyze(
        model_b3, data_2b, key_freqs_2b, 'freq_norm'
    )

    # --- 绘图 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    titles = [
        "Fig 2(b) Left: Time domain + Non-normalization",
        "Fig 2(b) Middle: Time domain + Normalization",
        "Fig 2(b) Right: Frequency domain + Normalization"
    ]
    plot_data_list = [plot_b1, plot_b2, plot_b3]
    
    # 确保plot_data中包含统一的频率轴参数
    for plot_data in plot_data_list:
        # 创建统一的频率轴参数
        max_freq = max(max(plot_data['freq_axis_input']), max(plot_data['freq_axis_pred']))
        plot_data['freq_axis_common'] = np.linspace(0, max_freq, 1000)
    
    for ax, title, plot_data in zip(axes, titles, plot_data_list):
        # 绘制预测窗口(PRED_LEN)的FFT
        ax.plot(plot_data['freq_axis_common'], np.interp(plot_data['freq_axis_common'], plot_data['freq_axis_pred'], plot_data['true_fft_mag']), 'g-', label='Ground Truth', linewidth=2)
        ax.plot(plot_data['freq_axis_common'], np.interp(plot_data['freq_axis_common'], plot_data['freq_axis_pred'], plot_data['pred_fft_mag']), 'r-', label='Forecasting', linewidth=2)
        # 绘制输入用于对比
        ax.plot(plot_data['freq_axis_common'], np.interp(plot_data['freq_axis_common'], plot_data['freq_axis_input'], plot_data['input_fft_mag']), 'b--', label='Input (Lookback)', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("F (Frequency Component k)")
        ax.set_xlim(0, max(key_freqs_2b) * 1.5)

    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    
    plt.tight_layout()
    plt.suptitle("Figure 2(b) Conceptual Reproduction: Debiasing Strategies", y=1.03)
    plt.show()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figure_2b.png'))


# --- 6. 运行复现 ---
if __name__ == "__main__":
    
    # 运行 Figure 2(a) 复现
    run_figure_2a()
    
    # # 运行 Figure 2(b) 复现
    # run_figure_2b()