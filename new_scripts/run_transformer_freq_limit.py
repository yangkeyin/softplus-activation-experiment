import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

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
PRED_LEN = 100    # 预测序列长度（信号点数）

# 固定时间轴
X_AXIS_PRED = np.linspace(0, 4 * np.pi, PRED_LEN)

# 频率配置
FREQS_TRAIN = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]  # 训练频率
FREQS_TEST = [1.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0]  # 测试频率

# 训练参数
EPOCHS = 50
BATCH_SIZE = 64
LR = 0.0001
N_SAMPLES_TRAIN = 5000
N_SAMPLES_VAL = 1000

# 输出目录
OUTPUT_DIR = './figures/Transformer_freq_limit'
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
    x_axis_pred: 预测的时间轴
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
        
        # 创建输入条件 [freq, phase]，在seq_len维度上重复
        x_sample = np.tile([[freq, phase]], (seq_len, 1))
        X.append(x_sample)
        
        # 生成目标信号 y = sin(freq * x + phase)
        y_sample = np.sin(freq * x_axis_pred + phase)[:, np.newaxis]
        Y.append(y_sample)
    
    # 转换为张量
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y), dtype=torch.float32)
    
    return TensorDataset(X_tensor, Y_tensor)


# --- 训练函数 ---
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    """
    训练模型并保存最佳模型
    """
    best_val_loss = float('inf')
    best_model_path = os.path.join(OUTPUT_DIR, 'best_freq_model.pth')
    
    train_losses = []
    val_losses = []
    
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
        train_losses.append(train_loss)
        
        # 验证
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
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'保存新的最佳模型，验证损失: {best_val_loss:.6f}')
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_losses, label='训练损失')
    plt.plot(range(1, epochs+1), val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('训练过程')
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curve.png'))
    plt.close()
    
    return best_model_path


# --- 可视化函数 ---
def plot_freq_mse_summary(results_mse, trained_freqs):
    """
    绘制频率-误差总结图
    """
    freqs = list(results_mse.keys())
    mses = list(results_mse.values())
    
    # 确定颜色
    colors = ['green' if freq in trained_freqs else 'red' for freq in freqs]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(freqs)), mses, color=colors)
    plt.xlabel('频率')
    plt.ylabel('均方误差 (MSE)')
    plt.yscale('log')
    plt.title('不同频率下的模型性能')
    plt.xticks(range(len(freqs)), freqs)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='训练过的频率'),
        Patch(facecolor='red', label='未见过的频率')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'freq_limit_mse_summary.png'))
    plt.close()

def plot_fits(freqs, y_tests, y_preds, mses, x_axis_pred):
    """
    绘制所有频率的拟合细节图
    """
    n_freqs = len(freqs)
    n_cols = 4
    n_rows = (n_freqs + n_cols - 1) // n_cols
    
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    
    for i, (freq, y_test, y_pred, mse) in enumerate(zip(freqs, y_tests, y_preds, mses)):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(x_axis_pred, y_test.flatten(), 'b-', label='真实信号')
        plt.plot(x_axis_pred, y_pred.flatten(), 'r--', label='预测信号')
        plt.xlabel('x')
        plt.ylabel('幅度')
        plt.title(f'Freq = {freq}, MSE = {mse:.6f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'freq_limit_fits.png'))
    plt.close()


# --- 主函数 ---
def main():
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
    best_model_path = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
    
    # 4. 评估模型
    print("开始评估模型...")
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    results_mse = {}
    y_tests_list = []
    y_preds_list = []
    mses_list = []
    
    for freq in FREQS_TEST:
        # 为每个频率生成测试数据（使用固定相位0）
        test_dataset = generate_data(1, [freq], SEQ_LEN, PRED_LEN, X_AXIS_PRED, fixed_phase=0)
        test_loader = DataLoader(test_dataset, batch_size=1)
        
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                Y_pred = model(X_batch)
                
                # 计算MSE
                mse = criterion(Y_pred, Y_batch).item()
                results_mse[freq] = mse
                
                # 保存结果用于可视化
                y_tests_list.append(Y_batch.cpu().numpy()[0])
                y_preds_list.append(Y_pred.cpu().numpy()[0])
                mses_list.append(mse)
                
                print(f'频率 {freq}, MSE: {mse:.6f}')
    
    # 5. 可视化结果
    print("生成可视化图表...")
    plot_freq_mse_summary(results_mse, FREQS_TRAIN)
    plot_fits(FREQS_TEST, y_tests_list, y_preds_list, mses_list, X_AXIS_PRED)
    
    print("实验完成！所有结果已保存到:", OUTPUT_DIR)

    plot_fits(FREQS_TEST, y_tests_list, y_preds_list, mses_list, X_AXIS_PRED)
    
    print("实验完成！所有结果已保存到:", OUTPUT_DIR)


if __name__ == "__main__":
    main()