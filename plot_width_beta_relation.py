import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, NullLocator

# 定义导数计算的光滑宽度
def get_width_from_derivative(beta):
    """
    计算导数计算的光滑宽度
    Args:
        beta (numpy.ndarray): Softplus 函数的参数
    Returns:
        numpy.ndarray: 导数计算的光滑宽度
    """
    # W = 2 * ln(9) / beta
    return (2 * np.log(9)) / beta

# 定义半宽高的光滑宽度
def get_width_from_half_corner(beta):
    """
    计算半宽高的光滑宽度
    Args:
        beta (numpy.ndarray): Softplus 函数的参数
    Returns:
        numpy.ndarray: 半宽高的光滑宽度
    """
    # W = |ln(sqrt(2) - 1)| / beta
    return np.abs(np.log(np.sqrt(2) - 1)) / beta

# 绘制半宽高与导数计算的光滑宽度的关系图
def plot_width_relation(beta):
    """
    绘制半宽高与导数计算的光滑宽度的关系图
    Args:
        beta (numpy.ndarray): Softplus 函数的参数
    """
    width_derivative = get_width_from_derivative(beta)
    width_half_corner = get_width_from_half_corner(beta)
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.plot(beta, width_derivative, label='Width from Derivative')
    ax.plot(beta, width_half_corner, label='Width from Half Corner')
    ax.set_title('Width Relation between Derivative and Half Corner')
    ax.set_xlabel('beta')
    ax.set_ylabel('Width')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xticks([0.1, 1, 3, 10])
    ax.set_yticks([0.1, 1, 3, 10])
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == '__main__':
    beta = np.linspace(0.1, 10, 50)
    plot_width_relation(beta)