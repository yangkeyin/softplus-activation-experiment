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
FREQS_TRAIN = [6, 8, 9, 11, 12, 14]
FREQS_TEST = [6, 7, 8, 9, 10, 11, 12, 13, 14]

SEQ_LEN = 500
PRED_LEN = 500
TOTAL_POINTS = 5000

RANGE_TRAIN_START = 0
RANGE_TRAIN_END = 3500
RANGE_TEST_START = 3500
RANGE_TEST_END = 5000

EPOCHS = 20
BATCH_SIZE = 32
LR = 0.01
WEIGHT_DECAY = 1e-5
SEEDS = [42]

VIS_EPOCHS = [1, 5, 10, 20, 30, 40, 50]
VIS_SAMPLE_INDICES = [0, 100, 200]
TEST_NOISE_LEVEL = 0.05  # Noise level for input during testing 

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f'./figures/Freq_Generalization_Exp_Noise{TEST_NOISE_LEVEL}_{TIMESTAMP}'
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
    'vis_sample_indices': VIS_SAMPLE_INDICES,
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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
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


def evaluate_on_range_freq(model, wave, freq, range_start, range_end, criterion, noise_level=0.05):
    model.eval()
    dataset = WaveSeq2SeqDataset(wave, range_start, range_end, SEQ_LEN, PRED_LEN)
    loader = DataLoader(dataset, batch_size=256)
    
    mse_list = []
    pred_dict = {}
    sample_count = 0
    
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)

            # Add noise to input for robustness testing
            noise = torch.randn_like(src) * noise_level
            src_noisy = src + noise
            pred = model(src_noisy)
            batch_mse = torch.mean((pred - tgt) ** 2, dim=(1, 2))
            mse_list.extend(batch_mse.cpu().numpy())
            
            for i in range(src.size(0)):
                if sample_count in VIS_SAMPLE_INDICES:
                    pred_dict[sample_count] = (src_noisy[i].cpu(), pred[i].cpu(), tgt[i].cpu())
                sample_count += 1
    
    return np.array(mse_list), pred_dict


def plot_epoch_samples(freq, range_name, epoch, mse_list, pred_dict):
    freq_dir = os.path.join(OUTPUT_DIR, range_name, f'{freq:.1f}Hz')
    os.makedirs(freq_dir, exist_ok=True)
    
    sample_indices = sorted([i for i in VIS_SAMPLE_INDICES if i in pred_dict])
    if not sample_indices:
        return
    
    num_samples = len(sample_indices)
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for row, sample_idx in enumerate(sample_indices):
        src, pred, tgt = pred_dict[sample_idx]
        src = src.numpy().flatten()
        pred = pred.numpy().flatten()
        tgt = tgt.numpy().flatten()
        
        # 计算样本的实际起始索引
        # sample_idx 是在数据集中的位置，对应的实际采样点索引为
        actual_start_idx = RANGE_TRAIN_START + sample_idx if range_name == 'train' else RANGE_TEST_START + sample_idx
        
        # Fitting plot
        ax1 = axes[row, 0]
        x_input = np.arange(actual_start_idx, actual_start_idx + SEQ_LEN)
        x_pred = np.arange(actual_start_idx + SEQ_LEN, actual_start_idx + SEQ_LEN + PRED_LEN)
        
        ax1.plot(x_input, src, 'k-', linewidth=2, label='Input', alpha=0.8)
        ax1.plot(x_pred, tgt, 'b-', linewidth=2, label='Ground Truth')
        ax1.plot(x_pred, pred, 'r--', linewidth=2, label='Prediction')
        ax1.axvline(x=actual_start_idx + SEQ_LEN, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax1.set_ylabel('Amplitude', fontsize=10)
        ax1.set_title(f'Sample {sample_idx}: Fitting (Start={actual_start_idx})', fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        # 显式设置x轴刻度，确保起始点可见
        ax1.set_xticks([actual_start_idx, actual_start_idx + SEQ_LEN, actual_start_idx + SEQ_LEN + PRED_LEN])
        
        # Error plot - 使用 plot 而不是 stem，使点相互连接
        ax2 = axes[row, 1]
        error = pred - tgt
        error_mse = np.mean(error ** 2)
        error_std = np.std(error)
        
        ax2.plot(x_pred, error, 'b-', linewidth=1.5, marker='o', markersize=4, label='Error')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_ylabel('Residual', fontsize=10)
        ax2.set_title(f'Sample {sample_idx}: Error (MSE={error_mse:.6f}, Std={error_std:.6f})', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        # 显式设置x轴刻度
        ax2.set_xticks([actual_start_idx + SEQ_LEN, actual_start_idx + SEQ_LEN + PRED_LEN])
    
    range_label = f'{range_name.upper()} Range ({RANGE_TRAIN_START}-{RANGE_TRAIN_END})' if range_name == 'train' else f'{range_name.upper()} Range ({RANGE_TEST_START}-{RANGE_TEST_END})'
    fig.suptitle(f'Freq={freq:.1f}Hz, Epoch={epoch}, {range_label}(Input Noise={TEST_NOISE_LEVEL})', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(freq_dir, f'epoch_{epoch:03d}_samples.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_mse_vs_epoch(all_epoch_mses, range_name):
    epochs_list = sorted(set(r['epoch'] for freq_data in all_epoch_mses.values() for r in freq_data))
    
    fig, ax = plt.subplots(figsize=(13, 8))
    
    for freq in FREQS_TEST:
        freq_data = all_epoch_mses.get(freq, [])
        if not freq_data:
            continue
        
        means, stds = [], []
        for epoch in epochs_list:
            for r in freq_data:
                if r['epoch'] == epoch:
                    means.append(np.mean(r['mse_list']))
                    stds.append(np.std(r['mse_list']))
                    break
        
        if not means:
            continue
        
        means, stds = np.array(means), np.array(stds)
        
        if freq in FREQS_TRAIN:
            linestyle, linewidth, alpha = '-', 2.5, 0.9
            label_suffix = ' (Train)'
        else:
            linestyle, linewidth, alpha = '--', 2.0, 0.7
            label_suffix = ' (Test)'
        
        ax.plot(epochs_list, means, linestyle=linestyle, linewidth=linewidth, alpha=alpha, marker='o', markersize=6, label=f'{freq:.1f}Hz{label_suffix}')
        lower = np.maximum(means - stds, 1e-8)
        upper = means + stds
        ax.fill_between(epochs_list, lower, upper, alpha=0.15)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_yscale('log')
    range_label = 'Train Range (0-3500)' if range_name == 'train' else 'Test Range (3500-5000)'
    # Set plot title with improved readability
    ax.set_title(f'MSE vs Epoch ({range_label})', fontsize=14, fontweight='bold')
    
    # Adjust legend for better clarity
    ax.legend(fontsize=10, loc='upper right', ncol=2)
    
    # Enhance grid visibility
    ax.grid(True, alpha=0.3, which='both')
    
    # Dynamically adjust y-axis limits based on data
    ax.set_ylim(1e-4, 1e-1)
    
    # Ensure layout is tight and directories exist
    plt.tight_layout()
    
    # 添加图片保存代码
    range_dir = os.path.join(OUTPUT_DIR, range_name)
    os.makedirs(range_dir, exist_ok=True)
    save_path = os.path.join(range_dir, f'mse_vs_epoch_{range_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  MSE vs Epoch plot saved to: {save_path}')
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
        optimizer = optim.Adam(model.parameters(), lr=LR)
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
                
                for range_name, range_start, range_end in [('test', RANGE_TEST_START, RANGE_TEST_END)]:
                    print(f"    Evaluating {range_name.upper()} Range:")
                    
                    for freq in FREQS_TEST:
                        mse_list, pred_dict = evaluate_on_range_freq(model, waves[freq], freq, range_start, range_end, criterion, noise_level=TEST_NOISE_LEVEL)
                        
                        epoch_mses[range_name][freq].append({'epoch': epoch, 'mse_list': mse_list, 'pred_dict': pred_dict})
                        
                        if len(pred_dict) > 0:
                            plot_epoch_samples(freq, range_name, epoch, mse_list, pred_dict)
                        
                        mean_mse = np.mean(mse_list)
                        std_mse = np.std(mse_list)
                        
                        gen_type = "(Train)" if freq in FREQS_TRAIN else ("(Extrap-Low)" if freq < min(FREQS_TRAIN) else ("(Extrap-High)" if freq > max(FREQS_TRAIN) else "(Interp)"))
                        print(f"      {freq:5.1f}Hz {gen_type:12s} | MSE: {mean_mse:.8f} ± {std_mse:.8f}")
                
                print()
        
        print(f"  Training completed for seed {seed}\n")
        
        for range_name in ['test']:
            for freq in FREQS_TEST:
                for result in epoch_mses[range_name][freq]:
                    all_epoch_mses[range_name][freq].append(result)
    
    print("[3/4] Generating MSE vs Epoch curves...\n")
    for range_name in ['test']:
        plot_mse_vs_epoch(all_epoch_mses[range_name], range_name)
    
    print("\n" + "=" * 100)
    print("Experiment completed!")
    print("=" * 100)
    print(f"\nResults saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
