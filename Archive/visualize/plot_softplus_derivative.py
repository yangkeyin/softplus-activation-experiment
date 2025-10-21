import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' # 解决多线程库冲突
import torch
import matplotlib.pyplot as plt
import numpy as np

# 定义输入范围
numpoints = 1000
x = torch.linspace(-4*np.pi, 4*np.pi, numpoints)  # 选择合适的范围以展示导数特性
# 选择一组合适的beta值
beta = [0.5, 1, 2, 5, 10, 20]  # 不同的beta值会影响曲线的陡峭程度
# 定义输出的目录
output_dir = 'softplus_activation'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 计算Softplus函数的导数
def softplus_derivative(x, beta=1.0):
    '''
    计算softplus函数的导数。
    d/dx Softplus(x) = 1 / (1 + e^(-beta*x))
    args:
        x(torch.Tensor): 输入张量
        beta(float): 缩放因子，默认值为1.0
    returns:
        softplus导数的值(torch.Tensor)
    '''
    return 1.0 / (1.0 + torch.exp(-beta * x))

# 绘制Softplus导数图像
def plot_softplus_derivative(x, beta):
    '''
    绘制softplus函数的导数图像。
    args:
        x(torch.Tensor): 输入张量
        beta(list of float): 缩放因子列表
    '''
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制不同beta值下的导数曲线
    for b in beta:
        y = softplus_derivative(x, b)
        ax.plot(x.numpy(), y.numpy(), label=f'beta={b}')
    
    # 添加参考线
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # 设置图像属性
    ax.set_xlabel('x')
    ax.set_ylabel('d/dx Softplus(x)')
    ax.set_title('Derivatives of Softplus Activation Functions with Different Beta Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 设置y轴范围以便更好地观察曲线
    ax.set_ylim(-0.05, 1.05)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f'{output_dir}/softplus_derivative_comparison.png', dpi=300)
    plt.show()

# 绘制特定点的导数对比图
def plot_derivative_comparison_at_point(beta_values, point_x=5):
    '''
    绘制在特定点x处，不同beta值对应的导数值对比。
    args:
        beta_values(list of float): 缩放因子列表
        point_x(float): 要计算导数的特定点
    '''
    x_tensor = torch.tensor([point_x])
    print(f"计算导数的点 x={x_tensor}")
    derivatives = [softplus_derivative(x_tensor, b).item() for b in beta_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([str(b) for b in beta_values], derivatives, color='skyblue')
    
    ax.set_xlabel('Beta Value')
    ax.set_ylabel(f'Derivative at x={point_x}')
    ax.set_title(f'Softplus Derivative at x={point_x} for Different Beta Values')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在每个柱子上标注数值
    for i, val in enumerate(derivatives):
        ax.text(i, val + 0.01, f'{val:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/softplus_derivative_at_x{point_x}.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    # 绘制不同beta值的导数曲线
    plot_softplus_derivative(x, beta)
    
    # 绘制在x=0处不同beta值的导数对比柱状图
    plot_derivative_comparison_at_point(beta, point_x=0.1)
    
    print("Softplus导数图像绘制完成！")
    print(f"图像已保存至 {output_dir} 目录")
