import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import copy

# 设置Matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# --- 配置 ---
# 确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

# 设置设备为GPU（如果可用）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {DEVICE}')

# 定义参数
SEQ_LEN = 10      # 输入序列长度（条件重复次数）
PRED_LEN = 200    # 预测序列长度（信号点数）
X_AXIS_PRED = np.linspace(0, 2 * np.pi, PRED_LEN)  # 固定的输出时间轴



# 频率配置
FREQS_TRAIN = [5.0, 10.0, 15.0]  # 训练频率
FREQS_TEST = [1.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0]  # 测试频率

# 训练参数
EPOCHS = 200
BATCH_SIZE = 64
LR = 0.0001
N_SAMPLES_TRAIN = 5000
N_SAMPLES_VAL = 1000
N_SAMPLES_TEST = 10  # 测试样本数量
SEEDS = [100, 200, 300, 400, 500]  # 多个随机种子
EVAL_EVERY_N_EPOCHS = 10  # 每10个epoch评估一次

# 输出目录
OUTPUT_DIR = './figures/Transformer_freq_limit_epoch200_gap10_predlen200_2pi_valpi'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 位置编码类 ---
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


# --- 简化的Transformer模型 ---
class SimpleTransformerModel(nn.Module):
    """一个简化的基于Transformer Encoder的预测模型"""
    def __init__(self, d_model=32, nhead=4, num_layers=2, input_dim=2, output_dim=1, seq_len=SEQ_LEN, pred_len=PRED_LEN):
        super(SimpleTransformerModel, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 输入嵌入层 (将频率和相位条件映射到d_model维度)
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer 编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 输出层 (将d_model维度映射回预测长度)
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


# --- 数据生成函数 ---
def generate_data(n_samples, freqs_list, seq_len, pred_len, x_axis_pred, fixed_phase=None):
    """
    生成条件生成任务的数据
    
    参数:
    n_samples: 样本数量
    freqs_list: 频率列表
    seq_len: 输入序列长度
    pred_len: 预测序列长度
    x_axis_pred: 固定的输出时间轴
    fixed_phase: 固定相位值（用于测试），默认为None
    
    返回:
    TensorDataset: 包含X和Y的数据集
    """
    X = []
    Y = []
    
    for _ in range(n_samples):
        # 随机选择频率
        freq = np.random.choice(freqs_list)
        
        # 生成相位
        if fixed_phase is not None:
            phase = fixed_phase
        else:
            phase = np.random.uniform(0, 2 * np.pi)
        
        # 创建输入条件 [freq, phase]，重复seq_len次
        x_sample = np.tile([freq, phase], (seq_len, 1))
        X.append(x_sample)
        
        # 生成目标信号 y = sin(freq * x_axis_pred + phase)
        y_sample = np.sin(freq * x_axis_pred + phase)[:, np.newaxis]
        Y.append(y_sample)
    
    # 转换为张量
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y), dtype=torch.float32)
    
    return TensorDataset(X_tensor, Y_tensor)


# --- 训练函数（修改为记录每个epoch的结果） ---
def train_model_with_tracking(model, train_loader, val_loader, test_loaders, criterion, optimizer, epochs, seed):
    """
    训练模型并记录每个epoch的结果
    """

    best_val_loss = float('inf')
    best_model_path = os.path.join(OUTPUT_DIR, f'best_freq_model_seed{seed}.pth')

    train_losses = []
    val_losses = []

    epoch_mse_dict = {}
    epoch_fit_dict = {}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)

            # 前向传播
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        
        # 验证
        if (epoch + 1) % EVAL_EVERY_N_EPOCHS == 0:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                    Y_pred = model(X_batch)
                    loss = criterion(Y_pred, Y_batch)
                    val_loss += loss.item() * X_batch.size(0)

            # 计算平均验证损失
            val_loss /= len(val_loader.dataset)
            
            # 每eval_step个epoch记录一次损失（包括第1和最后一个epoch）
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Seed {seed}, Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f'保存新的最佳模型，验证损失: {best_val_loss:.6f}')

            # 在每个epoch后评估所有测试频率
            epoch_mse_dict[epoch + 1] = {}
            epoch_fit_dict[epoch + 1] = {}

            for freq, test_loader in test_loaders.items():
                with torch.no_grad():
                    for X_batch, Y_batch in test_loader:
                        X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                        Y_pred = model(X_batch)

                        # 计算MSE
                        mse = criterion(Y_pred, Y_batch).item()
                        epoch_mse_dict[epoch + 1][freq] = mse
                        epoch_fit_dict[epoch + 1][freq] = (Y_batch.cpu().numpy()[0], Y_pred.cpu().numpy()[0])

    return train_losses, val_losses, epoch_mse_dict, epoch_fit_dict, best_model_path



# --- 新的可视化函数 ---
def plot_training_loss(all_train_losses, all_val_losses):
    """
    绘制训练损失图（平均和fillbetween）
    仅在每10个epoch（以及第1和最后1个epoch）处有数据点
    """
    # 转换为numpy数组
    train_losses_array = np.array(all_train_losses)  # (num_seeds, num_recorded_epochs)
    val_losses_array = np.array(all_val_losses)

    # 计算平均和标准差
    train_mean = np.mean(train_losses_array, axis=0)
    train_std = np.std(train_losses_array, axis=0)
    val_mean = np.mean(val_losses_array, axis=0)
    val_std = np.std(val_losses_array, axis=0)

    # 构建对应的epoch号
    # 因为eval_step = EVAL_EVERY_N_EPOCHS = 5，记录的epochs为: [5, 10, 15, ..., 50]
    num_recorded = len(train_mean)
    # 从 EVAL_EVERY_N_EPOCHS 开始，每隔 EVAL_EVERY_N_EPOCHS 记录一次
    epochs = [i * EVAL_EVERY_N_EPOCHS for i in range(1, num_recorded + 1)]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_mean, 'b-', label='训练损失 (平均)', marker='o', markersize=6)
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.3, color='blue')
    plt.plot(epochs, val_mean, 'r-', label='验证损失 (平均)', marker='s', markersize=6)
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.3, color='red')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.yscale('log')
    plt.legend(fontsize=11)
    plt.title('训练损失曲线 (多个种子平均)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss.png'), dpi=150)
    plt.close()


def plot_mse_vs_freq_over_epochs(all_epoch_mses, freqs_test, freqs_train):
    """
    绘制MSE vs Freq随epoch变化图
    适配新的 all_epoch_mses 结构：list[dict[epoch][freq] -> mse]
    """
    # 收集所有种子、所有 epoch、所有频率的 MSE
    # 先确定有哪些 epoch 被记录
    recorded_epochs = sorted(all_epoch_mses[0].keys())  # 所有种子记录相同的 epoch 列表
    n_epochs = len(recorded_epochs)
    n_freqs = len(freqs_test)
    n_seeds = len(all_epoch_mses)

    # 构造三维数组: (n_seeds, n_epochs, n_freqs)
    mse_array = np.zeros((n_seeds, n_epochs, n_freqs))
    for s, seed_mses in enumerate(all_epoch_mses):
        for e, epoch in enumerate(recorded_epochs):
            for f, freq in enumerate(freqs_test):
                mse_array[s, e, f] = seed_mses[epoch][freq]

    # 计算跨种子的均值
    mse_mean = np.mean(mse_array, axis=0)  # (n_epochs, n_freqs)

    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, n_epochs))

    plt.figure(figsize=(12, 8))
    for e, epoch in enumerate(recorded_epochs):
        plt.plot(freqs_test, mse_mean[e], 'o-',
                 color=colors[e],
                 label=f'Epoch {epoch}',
                 markersize=6, linewidth=2)

    # 标记训练频率
    has_train_freq_label = False
    for freq in freqs_train:
        if freq in freqs_test:
            plt.axvline(x=freq, color='green', linestyle='--', alpha=0.7,
                        label='训练频率' if not has_train_freq_label else "")
            has_train_freq_label = True

    # 设置X轴刻度只显示实际的测试频率
    plt.xticks(freqs_test, [f'{f:.1f}' for f in freqs_test], rotation=0)
    plt.xlabel('测试频率', fontsize=12)
    plt.ylabel('均方误差 (MSE)', fontsize=12)
    plt.yscale('log')
    plt.title('MSE vs 频率 随训练进展变化', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mse_vs_freq_over_epochs.png'), dpi=150)
    plt.close()


def plot_freq_fits_per_epoch(all_epoch_fits, all_epoch_mses, freqs_test, x_axis_pred):
    """
    为每个频率创建单独的文件夹，为每个记录的epoch绘制拟合图
    """
    # 迭代每个记录的epoch
    recorded_epochs = sorted(all_epoch_mses[0].keys())  # 从第一个种子的记录中获取所有epoch
    
    for epoch in recorded_epochs:
        actual_epoch = epoch
        
        for freq in freqs_test:
            # 创建频率文件夹
            freq_dir = os.path.join(OUTPUT_DIR, f'freq_{freq}')
            os.makedirs(freq_dir, exist_ok=True)
            
            # 收集所有种子的预测
            seed_tests = []
            seed_preds = []
            seed_mses = []

            for seed_idx in range(len(all_epoch_fits)):
                fit_data = all_epoch_fits[seed_idx][epoch][freq]
                mse_data = all_epoch_mses[seed_idx][epoch][freq]
                seed_tests.append(fit_data[0])
                seed_preds.append(fit_data[1])
                seed_mses.append(mse_data)

            # 计算平均
            y_test_mean = np.mean(np.array(seed_tests), axis=0)
            y_pred_mean = np.mean(np.array(seed_preds), axis=0)
            mse_mean = np.mean(seed_mses)

            # 绘制当前epoch的拟合图（1x2子图）
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # 子图1: 拟合图
            axes[0].plot(x_axis_pred, y_test_mean.flatten(), 'b-', label='真实信号', linewidth=2)
            axes[0].plot(x_axis_pred, y_pred_mean.flatten(), 'r--', label='预测信号', linewidth=2)
            axes[0].set_xlabel('X轴 (x_axis)', fontsize=11)
            axes[0].set_ylabel('幅度 (Amplitude)', fontsize=11)
            axes[0].set_title(f'拟合图', fontsize=12)
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)
            
            # 子图2: 误差图
            error = y_test_mean.flatten() - y_pred_mean.flatten()
            axes[1].plot(x_axis_pred, error, 'g-', linewidth=2)
            axes[1].set_xlabel('X轴 (x_axis)', fontsize=11)
            axes[1].set_ylabel('误差 (Error)', fontsize=11)
            axes[1].set_title(f'误差图', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            
            # 总标题
            fig.suptitle(f'Freq = {freq}, Epoch = {actual_epoch}, Avg MSE = {mse_mean:.6f}', 
                         fontsize=13, fontweight='bold')
            
            plt.tight_layout()
            save_path = os.path.join(freq_dir, f'epoch_{actual_epoch:03d}.png')
            plt.savefig(save_path, dpi=150)
            plt.close()


# --- 主函数 ---
def main():
    all_train_losses = []
    all_val_losses = []
    all_epoch_mses = []
    all_epoch_fits = []

    # 预生成所有测试数据
    test_loaders = {}
    for freq in FREQS_TEST:
        test_dataset = generate_data(N_SAMPLES_TEST, [freq], SEQ_LEN, PRED_LEN, X_AXIS_PRED, fixed_phase=0)
        test_loaders[freq] = DataLoader(test_dataset, batch_size=N_SAMPLES_TEST)

    for seed in SEEDS:
        print(f"\n=== 开始种子 {seed} 的实验 ===")

        # 设置种子
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 1. 数据准备
        print("生成训练和验证数据...")
        train_dataset = generate_data(N_SAMPLES_TRAIN, FREQS_TRAIN, SEQ_LEN, PRED_LEN, X_AXIS_PRED)
        val_dataset = generate_data(N_SAMPLES_VAL, FREQS_TRAIN, SEQ_LEN, PRED_LEN, X_AXIS_PRED)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # 2. 模型初始化
        model = SimpleTransformerModel(input_dim=2, output_dim=1, seq_len=SEQ_LEN, pred_len=PRED_LEN)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # 3. 训练模型
        print("开始训练模型...")
        train_losses, val_losses, epoch_mses, epoch_fits, best_model_path = train_model_with_tracking(
            model, train_loader, val_loader, test_loaders, criterion, optimizer, EPOCHS, seed
        )

        # 记录结果
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_epoch_mses.append(epoch_mses)
        all_epoch_fits.append(epoch_fits)

    # 4. 可视化结果
    print("\n生成可视化图表...")
    plot_training_loss(all_train_losses, all_val_losses)
    plot_mse_vs_freq_over_epochs(all_epoch_mses, FREQS_TEST, FREQS_TRAIN)
    plot_freq_fits_per_epoch(all_epoch_fits, all_epoch_mses, FREQS_TEST, X_AXIS_PRED)

    print("实验完成！所有结果已保存到:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
