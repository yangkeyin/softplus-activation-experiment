import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os
from src.utils import rescale, get_fq_coef
import torch.nn as nn

def train_and_evaluate(model, optimizer, X_train, y_train, X_test, y_test, config, beta, seed, results):
    """
    训练模型，评估并绘制损失曲线
    Args:
        model (nn.Module): 模型实例
        X_train (torch.Tensor): 训练集输入
        y_train (torch.Tensor): 训练集标签
        X_test (torch.Tensor): 测试集输入
        y_test (torch.Tensor): 测试集标签
        lr (float): 学习率
        epochs (int): 训练轮数
        n (int): 模型神经元数量
        beta (float): Softplus参数
        output_dir (str): 图像输出目录
        seed (int): 当前随机种子
        results (dict): 结果字典，用于存储中间结果
    Returns:
        std_dev (float): 测试集上的误差标准差
        y_pred (torch.Tensor): 模型在测试集上的预测值
    """
    criterion = nn.MSELoss()  # 均方误差损失函数
    
    train_losses = []
    val_losses = []
    
    epochs = config.EPOCHS_LIST[-1]
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train) # shape: (batch_size, 1)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # 每隔10轮记录一次损失和准确率
        if epoch % 10 == 0:
            train_losses.append(loss.item())
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test)
                val_losses.append(val_loss.item())

            print(f"Epoch [{epoch}/{epochs}], beta={beta}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

        # 如果当前轮次是epochlist中的一个元素
        if epoch + 1 in config.EPOCHS_LIST:
            # 计算当前轮次的预测值  
            model.eval()
            with torch.no_grad():
                # 记录预测值和误差标准差
                y_pred = model(X_test)
                error = y_test - y_pred
                std_dev = error.std().item()
                results['train_results'][epoch + 1][beta][seed]['y_pred'] = y_pred.cpu().numpy().flatten()
                results['train_results'][epoch + 1][beta][seed]['y_pred_std'] = std_dev

                # 记录当前轮次的频域
                if config.DATA_DIMENSION == 1:
                    f_ann = lambda x_norm: model(torch.FloatTensor(rescale(x_norm, config.DATA_RANGE)).reshape(-1, 1).to(config.DEVICE)).cpu().numpy().flatten()
                    ann_coeffs = get_fq_coef(f_ann)
                    results['train_results'][epoch + 1][beta][seed]['ann_coeffs'] = ann_coeffs
                else:
                    results['train_results'][epoch + 1][beta][seed]['ann_coeffs'] = None

    # --- 绘制并保存当前参数下的损失曲线 ---
    epochs_recorded = np.arange(len(train_losses)) * 10 
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.plot(epochs_recorded, train_losses, label='Train Loss')
    ax.plot(epochs_recorded, val_losses, label='Validation Loss')
    ax.set_title(f'Loss Curve for beta={beta}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    ax.set_yscale('log')
    
    # 修复网格线问题：分开设置主刻度和副刻度网格线样式
    ax.grid(True, which='major', linestyle='-', alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', alpha=0.5)
    
    # 修复副刻度显示数字的问题
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_minor_formatter(FormatStrFormatter('%.6g'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.6g'))
    ax.tick_params(axis='y', which='minor', labelsize=8, labelcolor='gray')

    ax.legend()
    loss_curve_path = os.path.join(config.OUTPUT_DIR, f'loss_curve_beta{beta}_seed{seed}.png')
    ax.figure.savefig(loss_curve_path)
    plt.close(ax.figure)
    
    # --- 在测试集上计算最终误差的标准差 ---
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        error = y_test - y_pred
        std_dev = error.std().item()
        
    return std_dev, y_pred