import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, NullLocator

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
# 设置负号正确显示
plt.rcParams["axes.unicode_minus"] = False

# 定义输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "optimizer_EPOCH10000")
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "comparisons")


def plot_all_loss_curves_comparison(results):
    """绘制所有优化器和学习率的损失曲线对比图"""
    if not os.path.exists(COMPARISON_DIR):
        os.makedirs(COMPARISON_DIR)
    
    # 创建训练损失对比图
    fig_train, ax_train = plt.subplots(figsize=(14, 10))
    
    # 创建验证损失对比图
    fig_val, ax_val = plt.subplots(figsize=(14, 10))
    
    # 为不同的优化器和学习率组合准备不同的颜色和线条样式
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyles = ['-', '--', '-.', ':']
    line_idx = 0
    
    # 遍历所有优化器和学习率
    for optimizer_name in results['optimizer_results'].keys():
        for lr in results['optimizer_results'][optimizer_name].keys():
            # 计算所有种子的平均训练损失和验证损失
            all_train_losses = []
            all_val_losses = []
            
            for seed in results['optimizer_results'][optimizer_name][lr].keys():
                all_train_losses.append(results['optimizer_results'][optimizer_name][lr][seed]['train_losses'])
                all_val_losses.append(results['optimizer_results'][optimizer_name][lr][seed]['val_losses'])
            
            # 计算平均值
            mean_train_losses = np.mean(all_train_losses, axis=0)
            mean_val_losses = np.mean(all_val_losses, axis=0)
            # 为了防止log(0)或log(负数)的错误，我们将所有非正数的值替换为一个非常小的正数
            # 比如 1e-8 (0.00000001)，这样既不影响图形的整体趋势，又能保证绘图成功。
            safe_mean_train_losses = np.clip(mean_train_losses, a_min=1e-8, a_max=None)
            safe_mean_val_losses = np.clip(mean_val_losses, a_min=1e-8, a_max=None)
            
            # 生成epochs序列
            epochs_recorded = np.arange(len(mean_train_losses)) * 10
            
            # 获取颜色和线条样式
            color = color_cycle[line_idx % len(color_cycle)]
            linestyle = linestyles[line_idx % len(linestyles)]
            line_idx += 1
            
            # 绘制训练损失曲线
            label = f"{optimizer_name}, lr={lr}"
            ax_train.plot(epochs_recorded, safe_mean_train_losses, label=label, color=color, linestyle=linestyle, linewidth=2)
            
            # 绘制验证损失曲线
            ax_val.plot(epochs_recorded, safe_mean_val_losses, label=label, color=color, linestyle=linestyle, linewidth=2)
    
    # 设置训练损失对比图的属性
    ax_train.set_title(f'所有优化器和学习率的训练损失曲线对比 (Neurons={results["fixed_params"]["n_neurons"]}, Beta={results["fixed_params"]["beta"]})')
    ax_train.set_xlabel('Epochs')
    ax_train.set_ylabel('MSE')
    ax_train.set_yscale('log')
    ax_train.legend(loc='best', ncol=2)
    ax_train.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, 'all_train_losses_comparison.png'))
    plt.close(fig_train)
    
    # 设置验证损失对比图的属性
    ax_val.set_title(f'所有优化器和学习率的验证损失曲线对比 (Neurons={results["fixed_params"]["n_neurons"]}, Beta={results["fixed_params"]["beta"]})')
    ax_val.set_xlabel('Epochs')
    ax_val.set_ylabel('MSE')
    ax_val.set_yscale('log')
    ax_val.legend(loc='best', ncol=2)
    ax_val.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, 'all_val_losses_comparison.png'))
    plt.close(fig_val)


def plot_optimizer_lr_grouped_loss_comparison(results):
    """按优化器分组绘制损失曲线对比图，便于比较相同优化器下不同学习率的效果"""
    if not os.path.exists(COMPARISON_DIR):
        os.makedirs(COMPARISON_DIR)
    
    # 为每个优化器创建单独的损失曲线对比图
    for optimizer_name in results['optimizer_results'].keys():
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        
        # 获取该优化器的所有学习率
        lrs = results['optimizer_results'][optimizer_name].keys()
        
        # 为不同的学习率准备不同的颜色
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        # 遍历所有学习率
        for i, lr in enumerate(lrs):
            # 计算所有种子的平均训练损失和验证损失
            all_train_losses = []
            all_val_losses = []
            
            for seed in results['optimizer_results'][optimizer_name][lr].keys():
                all_train_losses.append(results['optimizer_results'][optimizer_name][lr][seed]['train_losses'])
                all_val_losses.append(results['optimizer_results'][optimizer_name][lr][seed]['val_losses'])
            
            # 计算平均值
            mean_train_losses = np.mean(all_train_losses, axis=0)
            mean_val_losses = np.mean(all_val_losses, axis=0)
            
            # 生成epochs序列
            epochs_recorded = np.arange(len(mean_train_losses)) * 10
            
            # 获取颜色
            color = color_cycle[i % len(color_cycle)]
            
            # 绘制训练和验证损失曲线
            ax.plot(epochs_recorded, mean_train_losses, label=f'lr={lr} (Train)', color=color, linestyle='-', linewidth=2)
            ax.plot(epochs_recorded, mean_val_losses, label=f'lr={lr} (Val)', color=color, linestyle='--', linewidth=2)
        
        # 设置图表属性
        ax.set_title(f'{optimizer_name} 优化器不同学习率的损失曲线对比 (Neurons={results["fixed_params"]["n_neurons"]}, Beta={results["fixed_params"]["beta"]})')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('MSE')
        ax.set_yscale('log')
        ax.legend(loc='best')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(COMPARISON_DIR, f'{optimizer_name}_loss_comparison.png'))
        plt.close()

def plot_loss_comparison(results):
    """绘制不同优化器和学习率的最终误差标准差箱型图"""
    if not os.path.exists(COMPARISON_DIR):
        os.makedirs(COMPARISON_DIR)
    
    # 创建一个箱型图
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # 准备数据和标签
    data = []
    labels = []
    
    # 为每个优化器和学习率组合收集数据
    for optimizer_name in results['optimizer_results'].keys():
        for lr in results['optimizer_results'][optimizer_name].keys():
            # 收集所有种子的预测标准差
            std_devs = [results['optimizer_results'][optimizer_name][lr][seed]['y_pred_std']
                        for seed in results['optimizer_results'][optimizer_name][lr].keys()]
            data.append(std_devs)
            labels.append(f"{optimizer_name}, lr={lr}")
    
    # 绘制箱型图
    ax.boxplot(data, labels=labels, showfliers=True)
    
    ax.set_title('不同优化器和学习率的最终误差标准差箱型图')
    ax.set_ylabel('最终误差标准差')
    ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=90)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, 'optimizer_lr_std_comparison.png'))
    plt.close()

def plot_function_comparison(results):
    """绘制不同优化器和学习率的拟合函数对比图"""
    if not os.path.exists(COMPARISON_DIR):
        os.makedirs(COMPARISON_DIR)
    
    x_test = results['X_test']
    y_test = results['y_test']
    x_train = results['X_train']
    y_train = results['y_train']
    
    # 对x_test进行排序以便绘制连续曲线
    sort_indices = np.argsort(x_test.flatten())
    x_test_sorted = x_test[sort_indices]
    y_test_sorted = y_test[sort_indices]
    
    # 创建拟合函数对比图
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # 绘制真实函数和数据点
    ax.plot(x_test_sorted, y_test_sorted, 'k-', label='真实函数 sin(x)')
    ax.scatter(x_test, y_test, label='测试数据', color='green', s=10, alpha=0.5)
    ax.scatter(x_train, y_train, label='训练数据', color='black', marker='x', s=10, alpha=0.4)
    
    # 为每个优化器和学习率组合绘制平均预测曲线
    for optimizer_name in results['optimizer_results'].keys():
        for lr in results['optimizer_results'][optimizer_name].keys():
            # 计算所有种子的平均预测值
            y_preds = [results['optimizer_results'][optimizer_name][lr][seed]['y_pred'][sort_indices]
                      for seed in results['optimizer_results'][optimizer_name][lr].keys()]
            mean_y_pred = np.mean(y_preds, axis=0)
            
            # 计算平均预测的标准差
            std_devs = [results['optimizer_results'][optimizer_name][lr][seed]['y_pred_std']
                        for seed in results['optimizer_results'][optimizer_name][lr].keys()]
            mean_std_dev = np.mean(std_devs)
            
            label = f"{optimizer_name}, lr={lr} (std: {mean_std_dev:.6f})"
            line = ax.plot(x_test_sorted, mean_y_pred, label=label)[0]
            
            # 绘制平均预测值的点
            ax.scatter(x_test_sorted, mean_y_pred, s=10, alpha=0.5, color=line.get_color())
    
    ax.set_title(f'不同优化器和学习率的拟合函数对比 (Neurons={results["fixed_params"]["n_neurons"]}, Beta={results["fixed_params"]["beta"]}, Epochs={results["fixed_params"]["epochs"]})')
    ax.set_xlabel('x')
    ax.set_ylabel('预测值')
    ax.legend(loc='best')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, 'function_fitting_comparison.png'))
    plt.close()


def main():
    # 加载结果数据
    results_path = os.path.join(OUTPUT_DIR, 'optimizer_results.pkl')
    if not os.path.exists(results_path):
        print(f"错误: 结果文件 {results_path} 不存在。请先运行run.py。")
        return
    
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    print("开始生成优化器和学习率对比图...")
    
    # 绘制所有损失曲线对比图
    plot_all_loss_curves_comparison(results)
    
    # 按优化器分组绘制损失曲线对比图
    plot_optimizer_lr_grouped_loss_comparison(results)
    
    # 绘制拟合函数对比图
    plot_function_comparison(results)
    
    
    print("所有对比图已生成并保存到:", COMPARISON_DIR)


if __name__ == "__main__":
    main()