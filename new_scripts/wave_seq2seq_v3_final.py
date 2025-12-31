#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer Frequency Generalization Test v3
Protocol: Wave Seq2Seq Frequency Generalization Experiment
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}\n')

# Config
FREQS_TRAIN = [10.0, 20.0]
FREQS_TEST = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

SEQ_LEN = 100
PRED_LEN = 100
TOTAL_POINTS = 5000

RANGE_TRAIN_START = 0
RANGE_TRAIN_END = 3500
RANGE_TEST_START = 3500
RANGE_TEST_END = 5000

EPOCHS = 50
BATCH_SIZE = 32
LR = 0.001
WEIGHT_DECAY = 1e-5
SEEDS = [42]

VIS_EPOCHS = [1, 5, 10, 20, 30, 40, 50]

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f'./figures/Freq_Generalization_Exp_{TIMESTAMP}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

config = {
    'freqs_train': FREQS_TRAIN,
    'freqs_test': FREQS_TEST,
    'seq_len': SEQ_LEN,
    'pred_len': PRED_LEN,
    'total_points': TOTAL_POINTS,
    'range_train': [RANGE_TRAIN_START, RANGE_TRAIN_END],
    'range_test': [RANGE_TEST_START, RANGE_TEST_END],
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'lr': LR,
    'vis_epochs': VIS_EPOCHS,
}
with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)


def generate_wave_full(freq, total_points=TOTAL_POINTS):
    """Generate sine wave from [0, 2pi]"""
    x = np.linspace(0, 2 * np.pi, total_points)
    wave = np.sin(freq * x)
    return wave


class WaveSeq2SeqDataset(Dataset):
    """Wave Seq2Seq Dataset"""
    def __init__(self, wave, range_start, range_end, seq_len=SEQ_LEN, pred_len=PRED_LEN):
        self.wave = wave
        self.range_start = range_start
        self.range_end = range_end
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.max_start = range_end - seq_len - pred_len + 1
        self.min_start = range_start
        self.num_samples = max(0, self.max_start - self.min_start)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start_idx = self.min_start + idx
        seq = self.wave[start_idx:start_idx + self.seq_len]
        target = self.wave[start_idx + self.seq_len:start_idx + self.seq_len + self.pred_len]
        
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(-1)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(-1)
        
        return seq, target


class TransformerSeq2SeqModel(nn.Module):
    """Transformer Model"""
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_ff=128, seq_len=SEQ_LEN, pred_len=PRED_LEN):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.input_embed = nn.Linear(1, d_model)
        self.register_buffer('pos_encoding', self._create_pos_encoding(seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            batch_first=True, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(seq_len * d_model, pred_len)
        
        self.to(DEVICE)
    
    def _create_pos_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(100.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, src):
        x = self.input_embed(src)
        x = x + self.pos_encoding.unsqueeze(0)
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)
        out = self.output_proj(x)
        return out.unsqueeze(-1)


def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for src, tgt in train_loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        pred = model(src)
        loss = criterion(pred, tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * src.size(0)
    return total_loss / len(train_loader.dataset)


def evaluate_on_range_freq(model, wave, range_start, range_end):
    """
    使用所有重叠窗口进行评估，用于计算MSE
    
    Returns:
        mse_list: 每个样本的MSE列表
    """
    model.eval()
    dataset = WaveSeq2SeqDataset(wave, range_start, range_end, SEQ_LEN, PRED_LEN)
    loader = DataLoader(dataset, batch_size=256)
    
    mse_list = []
    
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            pred = model(src)
            batch_mse = torch.mean((pred - tgt) ** 2, dim=(1, 2))
            mse_list.extend(batch_mse.cpu().numpy())
    
    return np.array(mse_list)


def evaluate_on_range_freq_non_overlapping(model, wave, range_start, range_end):
    """
    使用不重叠的窗口进行预测，用于可视化拼接
    
    从range_start开始，每次取SEQ_LEN个点进行预测，预测下一个PRED_LEN个点
    然后跳过PRED_LEN个点，继续下一个预测
    这样保证预测段不重叠
    
    Returns:
        all_preds_flat: 所有不重叠预测的拼接 (总长度,)
        all_targets_flat: 对应的真实值 (总长度,)
    """
    model.eval()
    
    all_preds_list = []
    all_targets_list = []
    
    with torch.no_grad():
        # 从range_start开始，每次跳过PRED_LEN个点
        pos = range_start
        
        while pos + SEQ_LEN + PRED_LEN <= range_end:
            # 获取输入序列
            src = wave[pos:pos + SEQ_LEN]
            tgt = wave[pos + SEQ_LEN:pos + SEQ_LEN + PRED_LEN]
            
            src_tensor = torch.tensor(src, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)  # (1, SEQ_LEN, 1)
            
            # 预测
            pred = model(src_tensor)  # (1, PRED_LEN, 1)
            
            all_preds_list.append(pred.cpu().numpy().squeeze())  # (PRED_LEN,)
            all_targets_list.append(tgt)  # (PRED_LEN,)
            
            # 跳到下一个不重叠的位置
            pos += PRED_LEN
    
    # 拼接所有预测和真值
    all_preds_flat = np.concatenate(all_preds_list) if all_preds_list else np.array([])
    all_targets_flat = np.concatenate(all_targets_list) if all_targets_list else np.array([])
    
    return all_preds_flat, all_targets_flat


def plot_epoch_samples(freq, range_name, epoch, range_start, range_end, all_preds, all_targets, wave):
    """
    绘制整个range的完整预测序列（不重叠拼接）
    
    all_preds: 所有不重叠预测的拼接
    all_targets: 对应的真实值
    wave: 完整的波形数据
    """
    freq_dir = os.path.join(OUTPUT_DIR, range_name, f'{freq:.1f}Hz')
    os.makedirs(freq_dir, exist_ok=True)
    
    if len(all_preds) == 0:
        return
    
    # 预测范围的起始点：range_start + SEQ_LEN
    pred_start = range_start + SEQ_LEN
    pred_end = pred_start + len(all_preds)
    x_pred = np.arange(pred_start, pred_end)
    
    # 获取对应的真实波形数据
    true_wave_pred = wave[pred_start:pred_end]
    
    # 创建一行两列的图
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # 左列: 拟合图
    ax1 = axes[0]
    ax1.plot(x_pred, true_wave_pred, 'r-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax1.plot(x_pred, all_preds, 'b--', linewidth=2, label='Prediction', alpha=0.8)
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title(f'Freq={freq:.1f}Hz - Fitting (Range: {range_start}-{range_end})', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 右列: 误差图
    ax2 = axes[1]
    error = all_preds - all_targets
    error_mse = np.mean(error ** 2)
    error_std = np.std(error)
    
    ax2.plot(x_pred, error, 'orange', linewidth=1.5, label='Prediction Error', alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.fill_between(x_pred, error, 0, alpha=0.3, color='orange')
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_ylabel('Error (Pred - Truth)', fontsize=12)
    ax2.set_title(f'Freq={freq:.1f}Hz - Error (MSE={error_mse:.6f}, Std={error_std:.6f})', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    save_path = os.path.join(freq_dir, f'epoch_{epoch:03d}_complete_range.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_mse_vs_epoch(all_epoch_mses, range_name):
    # all_epoch_mses: {freq: [{epoch: ..., mse_list: ...}, ...]}
    epochs_list = sorted(set(r['epoch'] for freq_data in all_epoch_mses.values() for r in freq_data))
    
    if not epochs_list:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for freq in FREQS_TEST:
        freq_data = all_epoch_mses.get(freq, [])
        if not freq_data:
            continue
        
        # 为每个epoch构建means和stds
        epoch_to_results = {r['epoch']: r for r in freq_data}
        
        means, stds = [], []
        valid_epochs = []
        
        for epoch in epochs_list:
            if epoch in epoch_to_results:
                mse_list = epoch_to_results[epoch]['mse_list']
                means.append(np.mean(mse_list))
                stds.append(np.std(mse_list))
                valid_epochs.append(epoch)
        
        if not means:
            continue
        
        means, stds = np.array(means), np.array(stds)
        valid_epochs = np.array(valid_epochs)
        
        if freq in FREQS_TRAIN:
            linestyle, linewidth, alpha = '-', 2.5, 0.9
            label_suffix = ' (Train)'
        else:
            linestyle, linewidth, alpha = '--', 2.0, 0.7
            label_suffix = ' (Test)'
        
        ax.plot(valid_epochs, means, linestyle=linestyle, linewidth=linewidth, alpha=alpha, marker='o', markersize=7, label=f'{freq:.1f}Hz{label_suffix}')
        lower = np.maximum(means - stds, 1e-8)
        upper = means + stds
        ax.fill_between(valid_epochs, lower, upper, alpha=0.15)
    
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('MSE Loss', fontsize=13)
    ax.set_yscale('log')
    range_label = 'Train Range (0-3500)' if range_name == 'train' else 'Test Range (3500-5000)'
    ax.set_title(f'MSE vs Epoch ({range_label})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', ncol=2)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    
    plt.tight_layout()
    os.makedirs(os.path.join(OUTPUT_DIR, range_name), exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, range_name, f'mse_vs_epoch_{range_name}_range.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {range_name}/mse_vs_epoch_{range_name}_range.png")


def main():
    print("\n" + "=" * 100)
    print("Transformer Frequency Generalization Test v3")
    print("=" * 100 + "\n")
    
    print("[1/4] Generating waveforms...")
    waves = {}
    for freq in FREQS_TEST:
        waves[freq] = generate_wave_full(freq, TOTAL_POINTS)
    print(f"  Generated {len(FREQS_TEST)} frequencies, {TOTAL_POINTS} points each\n")
    
    all_epoch_mses = defaultdict(lambda: defaultdict(list))
    
    print("[2/4] Training...\n")
    
    for seed_idx, seed in enumerate(SEEDS):
        print("=" * 100)
        print(f"Seed {seed_idx + 1}/{len(SEEDS)}: Seed={seed}")
        print("=" * 100)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        print("\nCreating training dataset...")
        train_datasets = []
        for freq in FREQS_TRAIN:
            dataset = WaveSeq2SeqDataset(waves[freq], RANGE_TRAIN_START, RANGE_TRAIN_END, SEQ_LEN, PRED_LEN)
            train_datasets.append(dataset)
            print(f"  {freq:.1f}Hz: {len(dataset)} samples")
        
        train_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"  Total: {len(train_dataset)} samples\n")
        
        model = TransformerSeq2SeqModel(d_model=32, nhead=4, num_layers=2)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        print("Training...\n")
        epoch_mses = defaultdict(lambda: defaultdict(list))
        
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d}/{EPOCHS} | Train Loss: {train_loss:.8f}")
            
            if epoch in VIS_EPOCHS:
                print(f"\n  -> Epoch {epoch} evaluation...")
                
                for range_name, range_start, range_end in [('train', RANGE_TRAIN_START, RANGE_TRAIN_END), ('test', RANGE_TEST_START, RANGE_TEST_END)]:
                    print(f"    Evaluating {range_name.upper()} Range:")
                    
                    for freq in FREQS_TEST:
                        # 计算MSE（使用所有重叠窗口）
                        mse_list = evaluate_on_range_freq(model, waves[freq], range_start, range_end)
                        epoch_mses[range_name][freq].append({'epoch': epoch, 'mse_list': mse_list})
                        
                        # 获取不重叠预测用于可视化
                        all_preds, all_targets = evaluate_on_range_freq_non_overlapping(model, waves[freq], range_start, range_end)
                        
                        if len(all_preds) > 0:
                            # 绘制完整range的可视化
                            plot_epoch_samples(freq, range_name, epoch, range_start, range_end, all_preds, all_targets, waves[freq])
                            
                            mean_mse = np.mean(mse_list)
                            std_mse = np.std(mse_list)
                            
                            gen_type = "(Train)" if freq in FREQS_TRAIN else ("(Extrap-Low)" if freq < min(FREQS_TRAIN) else ("(Extrap-High)" if freq > max(FREQS_TRAIN) else "(Interp)"))
                            print(f"      {freq:5.1f}Hz {gen_type:12s} | MSE: {mean_mse:.8f} ± {std_mse:.8f}")
                
                print()
        
        print(f"  Training completed for seed {seed}\n")
        
        for range_name in ['train', 'test']:
            for freq in FREQS_TEST:
                for result in epoch_mses[range_name][freq]:
                    all_epoch_mses[range_name][freq].append(result)
    
    print("[3/4] Generating MSE vs Epoch curves...\n")
    for range_name in ['train', 'test']:
        plot_mse_vs_epoch(all_epoch_mses[range_name], range_name)
    
    print("\n" + "=" * 100)
    print("Experiment completed!")
    print("=" * 100)
    print(f"\nResults saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
