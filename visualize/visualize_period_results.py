import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, NullLocator
import math

# ==============================================================================
# 配置参数
# ==============================================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../results/period/1017_SmallPoint_WidePeriod_800FixedTrainPoints")

# ==============================================================================
# 周期分析函数
# ==============================================================================
def run_period_analysis(results):
    """
    执行以周期为横坐标的分析和绘图。
    
    Args:
        results (dict): 包含实验结果的字典
    """
    print("Running in PERIOD analysis mode...")
    
    # 获取固定参数
    fixed_beta = results['fixed_beta']
    fixed_width = results['fixed_width']
    period_list = results['period_list']
    
    # 创建周期对应的标签
    pi_labels = [f'{period/np.pi:.2f}π' for period in period_list]
    
    # 获取实验结果
    period_results = results['period_results']
    
    # 为每个epoch绘制结果
    for epoch in period_results.keys():
        epoch_dir = os.path.join(OUTPUT_DIR, f'epoch_{epoch}')
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        
        # 绘制周期比较图
        fig_period, ax_period = plt.subplots(figsize=(12, 8))
        all_mean_stds = []
        
        # 为每个神经元数量绘制结果
        for n in period_results[epoch][next(iter(period_results[epoch].keys()))].keys():
            mean_stds = []
            # 为每个周期获取平均标准差
            for period in period_list:
                # 获取当前周期下所有种子的结果
                std_devs = [period_results[epoch][period][n][seed]['y_pred_std'] 
                           for seed in period_results[epoch][period][n].keys()]
                mean_std = np.mean(std_devs)
                mean_stds.append(mean_std)
                all_mean_stds.append(mean_std)
                
                # 为每个周期和神经元数量创建单独的预测图
                create_prediction_plot(results, epoch, period, n, fixed_beta, epoch_dir)
        
            # 绘制神经元数量n的周期-标准差曲线
            ax_period.plot(period_list, mean_stds, 'o-', label=f'n={n}')
        
        # 配置周期比较图
        ax_period.set_title(f'Mean Error Std Dev vs. Period (Epoch: {epoch}, β={fixed_beta:.4f})')
        ax_period.set_xlabel('Period (T)')
        ax_period.set_ylabel('Mean Standard Deviation of Error')
        ax_period.set_yscale('log')
        
        # 设置x轴刻度和标签
        ax_period.set_xticks(period_list)
        ax_period.set_xticklabels(pi_labels, rotation=45, ha='right')
        
        # 设置y轴刻度
        ax_period.set_yticks(sorted(list(set(all_mean_stds))))
        ax_period.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax_period.yaxis.set_minor_locator(NullLocator())
        
        ax_period.legend()
        ax_period.grid(True, which="both", linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, f'std_vs_period_comparison_{epoch}.png'))
        plt.close(fig_period)

# ==============================================================================
# 预测结果绘图函数
# ==============================================================================
def create_prediction_plot(results, epoch, period, n, fixed_beta, epoch_dir):
    """
    为特定周期、神经元数量和epoch创建预测结果图和误差图。
    
    Args:
        results (dict): 包含实验结果的字典
        epoch (int): 当前epoch
        period (float): 当前周期
        n (int): 神经元数量
        fixed_beta (float): 固定的beta值
        epoch_dir (str): 图像保存目录
    """
    k = 2 * np.pi / period
    period_in_pi = period / np.pi
    
    # 获取当前period值的测试数据（核心修改：使用period作为键）
    X_test_key = f'X_test_period_{period}'
    y_test_key = f'y_test_period_{period}'
    X_train_key = f'X_train_period_{period}'
    y_train_key = f'y_train_period_{period}'
    
    if X_test_key not in results or y_test_key not in results:
        print(f"Warning: Test data for period={period_in_pi:.2f}π not found in results.")
        return
        
    # 获取测试数据
    x_test = results[X_test_key]
    y_test = results[y_test_key]
    
    # 排序测试数据以便绘图
    sort_indices = np.argsort(x_test.flatten())
    x_test_sorted = x_test[sort_indices]
    y_test_sorted = y_test[sort_indices]

    # 获取训练数据
    X_train = results[X_train_key]
    y_train = results[y_train_key]
    # 排序训练数据
    sort_indices_train = np.argsort(X_train.flatten())
    X_train_sorted = X_train[sort_indices_train]
    y_train_sorted = y_train[sort_indices_train]
    
    # 创建预测图
    fig_fit, ax_fit = plt.subplots(figsize=(10, 6))

    # 绘制训练数据点的预测曲线
    ax_fit.plot(X_train_sorted, y_train_sorted, label=f'Trainline sin({k:.2f}x) [Period={period_in_pi:.2f}π]')

    # 绘制训练数据点
    ax_fit.scatter(X_train_sorted, y_train_sorted, label='Train Data', color='green', s=20, alpha=0.7)
    # 从周期计算频率并显示
    ax_fit.plot(x_test_sorted, y_test_sorted, label=f'Testline')
    
    # 绘制测试数据点
    ax_fit.scatter(x_test_sorted, y_test_sorted, label='Test Data', color='blue', s=10, alpha=0.5)
    

    
    # 创建误差图
    fig_error, ax_error = plt.subplots(figsize=(10, 6))
    
    # 获取所有种子的预测结果并计算平均值
    all_y_preds = []
    for seed in results['period_results'][epoch][period][n].keys():
        y_pred = results['period_results'][epoch][period][n][seed]['y_pred']
        if y_pred is not None:
            # 对预测结果进行排序（与测试数据对应）
            y_pred_sorted = y_pred[sort_indices]
            all_y_preds.append(y_pred_sorted)
            
            # 在误差图上绘制单个种子的误差
            error = y_pred_sorted - y_test_sorted.flatten()
            ax_error.plot(x_test_sorted, error, alpha=0.3, label=f'seed={seed}')
    
    # 计算平均预测值和平均误差
    if all_y_preds:
        mean_y_pred = np.mean(all_y_preds, axis=0)
        mean_error = np.mean([y_pred - y_test_sorted.flatten() for y_pred in all_y_preds], axis=0)
        
        # 在预测图上绘制平均预测值
        line = ax_fit.plot(x_test_sorted, mean_y_pred, 'r-', linewidth=2)[0]
        ax_fit.scatter(x_test_sorted, mean_y_pred, s=10, alpha=0.5, color='red')
        
        # 在误差图上绘制平均误差
        ax_error.plot(x_test_sorted, mean_error, 'r-', linewidth=2, label='Mean Error')
    
    # 配置预测图
    ax_fit.set_title(f'Neuron {n} - Period {period_in_pi:.2f}π (k={k:.2f}), Epoch {epoch}')
    ax_fit.set_xlabel('x')
    ax_fit.set_ylabel('Predicted Value')
    ax_fit.set_ylim(-1.2, 1.2)  # 设置y轴范围以更好地显示sin函数
    ax_fit.legend()
    
    # 创建周期特定的目录
    period_dir = os.path.join(epoch_dir, f'period_{period_in_pi:.2f}pi')
    if not os.path.exists(period_dir):
        os.makedirs(period_dir)
    
    # 保存预测图
    fig_fit.savefig(os.path.join(period_dir, f'neuron_{n}_prediction.png'))
    plt.close(fig_fit)
    
    # 配置误差图
    ax_error.set_title(f'Error for Neuron {n} - Period {period_in_pi:.2f}π (k={k:.2f}), Epoch {epoch}')
    ax_error.set_xlabel('Test x')
    ax_error.set_ylabel('Error')
    ax_error.legend()
    
    # 保存误差图
    fig_error.savefig(os.path.join(period_dir, f'neuron_{n}_error.png'))
    plt.close(fig_error)

# ==============================================================================
# 主函数
# ==============================================================================
if __name__ == '__main__':
    # 加载结果数据
    results_file = os.path.join(OUTPUT_DIR, 'results.pkl')
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        print("Please run run_period_experiment.py first to generate the results.")
        exit(1)
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    # 执行周期分析
    run_period_analysis(results)
    
    print(f"\n{'='*50}")
    print(f"可视化完成！结果已保存到 {OUTPUT_DIR}")
    print(f"{'='*50}")