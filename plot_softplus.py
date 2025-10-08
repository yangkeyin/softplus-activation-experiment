
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' # 解决多线程库冲突
import torch
import matplotlib.pyplot as plt
import numpy as np



# 定义softplus函数的输入范围
numpoints = 1000
x = torch.linspace(-4*np.pi, 4*np.pi, numpoints)
# 选择一组合适的beta和threshold值
beta = [2.24,1.12,0.56,0.28,0.19,0.14,0.09,0.07]
threshold = [20,40,80,160]
# 定义输出的目录
output_dir = 'softplus_activation'  
# 检查输出目录是否存在，若不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 编写softplus的数学公式
def softplus(x, beta=1.0, threshold=20.0):
    '''
    计算softplus函数的值。
    softplus(x) = 1 / beta * log(1 + exp(x * beta))
    当x很大时，直接返回x，避免数值溢出。
    args:
        x(torch.Tensor): 输入张量
        beta(float): 缩放因子，默认值为1.0
        threshold(float): 阈值，默认值为20.0
    returns:
        softplus(x) 的值(torch.Tensor): softplus函数在输入张量x上的输出值
    '''
    # 为了数值稳定性，当beta * x超过threshold时，直接返回x
    mask = (beta * x > threshold).float() # 返回0，1的掩膜向量
    return mask * x + (1 - mask) * (1 / beta * torch.log(1 + torch.exp(x * beta)))


def plot_softplus_beta(x, beta, threshold):
    '''
    绘制softplus函数的beta对比图像。
    args:
        beta(list of float): 缩放因子列表
        threshold(list of float): 阈值列表
        numpoints(int): 输入范围的点数，默认值为1000
    '''
    # 计算softplus函数的值
    for t in threshold:
        fig, ax = plt.subplots(figsize = (16, 8))
        for b in beta:
            y = softplus(x, b, t)
            ax.plot(x.numpy(), y.numpy(), label=f'beta={b}')
        # 绘制图像
        ax.set_xlabel('x')
        ax.set_ylabel('softplus(x)')
        ax.set_title(F'Comparison of Softplus Activation Functions with Different Beta Values (threshold={t})')
        ax.legend()
        ax.grid(True)
    
        # 将生成的图像保存至softplus_activation目录下，命名为softplus_beta_comparison_threshold.png
        plt.savefig(f'{output_dir}/softplus_beta_comparison_{t}.png')
        plt.show()

# 绘制threshold对比图像
def plot_softplus_threshold(x, beta, threshold):
    '''
    绘制softplus函数的threshold对比图像。
    args:
        beta(float): 缩放因子
        threshold(list of float): 阈值列表
        numpoints(int): 输入范围的点数，默认值为1000
    '''
    # 计算softplus函数的值
    fig, ax = plt.subplots(figsize = (16, 8))
    for t in threshold:
        y = softplus(x, beta, t)
        ax.plot(x.numpy(), y.numpy(), label=f'threshold={t}')
    # 绘制图像
    ax.set_xlabel('x')
    ax.set_ylabel('softplus(x)')
    ax.set_title(F'Comparison of Softplus Activation Functions with Different Threshold Values (beta={beta})')
    ax.legend()
    ax.grid(True)

    # 将生成的图像保存至softplus_activation目录下，命名为softplus_threshold_comparison_beta.png
    plt.savefig(f'{output_dir}/softplus_threshold_comparison_{beta}.png')
    plt.show()

if __name__ == '__main__':
    plot_softplus_beta(x, beta, threshold)
    plot_softplus_threshold(x, 10, threshold)
