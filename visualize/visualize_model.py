import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter

# ==============================================================================
# 1. 配置参数 (Configuration)
# ==============================================================================
# 输入的 pickle 文件路径
RESULTS_PKL_PATH = os.path.join(os.path.dirname(__file__), "../results/demo/results.pkl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../results/demo/analysis_plots")
# ==============================================================================
# 2. 绘图函数 (全部更新)
# ==============================================================================

def get_color_map(betas):
    """
    为每个 (优化器, 学习率) 组合生成一个唯一的颜色映射。
    """
    # 使用 matplotlib 的 tab10 颜色方案
    color_cycle = plt.cm.tab10(np.linspace(0, 1, len(betas)))
    color_map = {beta: color for beta, color in zip(betas, color_cycle)}
    return color_map


def plot_fitting_function_comparison(results, base_output_dir, color_map):
    """
    为每个epoch绘制一张汇总的拟合函数对比图。
    图中包含所有优化器和学习率组合的结果。
    不同seed的结果用浅色表示，其平均值用深色实线表示。
    """
    print("Generating summarized fitting function comparison plots...")
    X_test = results['X_test']
    y_test = results['y_test']
    sorted_indices = np.argsort(X_test.flatten())
    X_test_sorted = X_test[sorted_indices]
    y_test_sorted = y_test[sorted_indices]

    train_results = results['train_results']

    # 按 Epoch 循环，为每个 Epoch 生成一张图
    for epoch_val, train_data in train_results.items():
        epoch_dir = os.path.join(base_output_dir, f"epoch_{epoch_val}")
        os.makedirs(epoch_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

        # 绘制一次真实函数和误差参考线
        ax1.plot(X_test_sorted, y_test_sorted, 'k--', linewidth=2, label='True Function (sin(x))')
        ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)

        # 遍历该 Epoch 下的所有实验配置
        for beta, seed_data in train_data.items():
            color = color_map[beta]
            
            # 收集当前配置下所有 seed 的数据
            preds_from_seeds = []
            errors_from_seeds = []
            for seed, seed_result in seed_data.items():
                y_pred_sorted = seed_result['y_pred'][sorted_indices]
                preds_from_seeds.append(y_pred_sorted)
                errors_from_seeds.append(y_test_sorted.flatten() - y_pred_sorted.flatten())
            
            if not preds_from_seeds:
                continue

            # 绘制每个 seed 的浅色背景线
            for i in range(len(preds_from_seeds)):
                ax1.plot(X_test_sorted, preds_from_seeds[i], color=color, alpha=0.2)
                ax2.plot(X_test_sorted, errors_from_seeds[i], color=color, alpha=0.2)
            
            # 计算平均值并绘制深色实线
            if len(preds_from_seeds) > 0:
                mean_pred = np.mean(preds_from_seeds, axis=0)
                mean_error = np.mean(errors_from_seeds, axis=0)
                
                label_avg = f'Avg - Beta={beta}'

                ax1.plot(X_test_sorted, mean_pred, color=color, linewidth=2.5, label=label_avg)
                ax2.plot(X_test_sorted, mean_error, color=color, linewidth=2.5)

        # 设置图像属性并保存
        ax1.set_ylabel('Value')
        ax1.set_title(f'Aggregated Fit Comparison\nEpoch: {epoch_val}')
        ax1.legend(loc='upper right')
        ax1.grid(True, which="both", ls="--", linewidth=0.5)
        # 展示副坐标轴的信息


        ax2.set_xlabel('x')
        ax2.set_ylabel('Prediction Error (True - Pred)')
        ax2.grid(True, which="both", ls="--", linewidth=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        filename = f"fit_comparison_summary_epoch_{epoch_val}.png"
        filepath = os.path.join(epoch_dir, filename)
        plt.savefig(filepath, dpi=150)
        plt.close(fig)
        
    print("Done.")


def plot_fitting_function_frequency_comparison(results, base_output_dir, color_map):
    """
    为每个epoch绘制一张汇总的频域对比图。
    (此函数已在上一轮修改，此处保持更新后的版本)
    """
    print("Generating summarized frequency domain comparison plots...")
    true_coeffs = results['true_coeffs']
    k = np.arange(len(true_coeffs))
    train_results = results['train_results']

    for epoch_val, train_data in train_results.items():
        epoch_dir = os.path.join(base_output_dir, f"epoch_{epoch_val}")
        os.makedirs(epoch_dir, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        ax1.semilogy(k, np.abs(true_coeffs), 'k--', linewidth=2, label='True Coefficients (sin(x))')

        for beta, seed_data in train_data.items():
            color = color_map[beta]
            coeffs_from_seeds = []
            errors_from_seeds = []
            for i, seed_result in enumerate(seed_data.values()):
                ann_coeffs = seed_result['ann_coeffs']
                coeffs_from_seeds.append(np.abs(ann_coeffs))
                errors_from_seeds.append(np.abs(true_coeffs - ann_coeffs))
                if not coeffs_from_seeds: continue
                # 绘制每个 seed 的浅色背景线
                ax1.semilogy(k, coeffs_from_seeds[i], color=color, alpha=0.2)
                ax2.semilogy(k, errors_from_seeds[i], color=color, alpha=0.2)
            mean_coeffs = np.mean(coeffs_from_seeds, axis=0)
            mean_error = np.mean(errors_from_seeds, axis=0)
            label_avg = f'Avg - Beta={beta}'
            ax1.semilogy(k, mean_coeffs, color=color, linewidth=2.5, label=label_avg)
            ax2.semilogy(k, mean_error, color=color, linewidth=2.5)

        ax1.set_ylabel('|Coefficient| (log scale)')
        ax1.set_title(f'Aggregated Frequency Domain Comparison\nEpoch: {epoch_val}')
        ax1.legend(loc='upper right')
        ax1.grid(True, which="both", ls="--", linewidth=0.5)
        ax2.set_xlabel('Frequency Index (k)')
        ax2.set_ylabel('|Coefficient Error| (log scale)')
        ax2.grid(True, which="both", ls="--", linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename = f"freq_comparison_summary_epoch_{epoch_val}.png"
        filepath = os.path.join(epoch_dir, filename)
        plt.savefig(filepath, dpi=150)
        plt.close(fig)
    print("Done.")


def plot_all_std_dev_comparison(results, base_output_dir, color_map):
    """
    绘制一张折线图，展示不同实验配置的误差标准差随epoch的变化趋势。
    """
    print("Generating standard deviation evolution plot...")
    
    train_results = results['train_results']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    all_epochs = sorted(list(train_results.keys()))

    # 遍历每种 (优化器, 学习率) 组合
    for beta, color in color_map.items():
        
        mean_stds_over_epochs = []
        std_of_stds_over_epochs = []
        valid_epochs = []

        # 遍历所有epoch，为当前配置收集数据
        for epoch in all_epochs:
            if beta in train_results[epoch]:
                seed_data = train_results[epoch][beta]
                stds_for_current_config = [res['y_pred_std'] for res in seed_data.values()]
                
                if stds_for_current_config:
                    valid_epochs.append(epoch)
                    mean_stds_over_epochs.append(np.mean(stds_for_current_config))
                    std_of_stds_over_epochs.append(np.std(stds_for_current_config))
        
        if not valid_epochs:
            continue
            
        mean_stds_over_epochs = np.array(mean_stds_over_epochs)
        std_of_stds_over_epochs = np.array(std_of_stds_over_epochs)
        
        # 绘制平均值折线
        label = f'Beta={beta}'
        ax.plot(valid_epochs, mean_stds_over_epochs, marker='o', linestyle='-', color=color, label=label)
        
        # 使用 fill_between 展示稳定性（标准差范围）
        ax.fill_between(valid_epochs, 
                        mean_stds_over_epochs - std_of_stds_over_epochs,
                        mean_stds_over_epochs + std_of_stds_over_epochs,
                        color=color, alpha=0.2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean std(y_true - y_pred) (log scale)')
    ax.set_title('Evolution of Prediction Error Standard Deviation')
    ax.set_yscale('log')
    ax.set_xticks(all_epochs) # 确保所有epoch点都在x轴上显示
    ax.legend()
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    
    plt.tight_layout()
    filepath = os.path.join(base_output_dir, "std_dev_evolution.png")
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print("Done.")


# ==============================================================================
# 3. 主执行流程
# ==============================================================================
def main():
    """
    主函数，加载数据并调用所有更新后的绘图函数。
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        with open(RESULTS_PKL_PATH, 'rb') as f:
            results = pickle.load(f)
        print(f"Successfully loaded results from '{RESULTS_PKL_PATH}'")
    except FileNotFoundError:
        print(f"Error: Results file not found at '{RESULTS_PKL_PATH}'")
        return
        
    # 首先生成颜色映射，确保所有图表颜色一致
    color_map = get_color_map(results['fixed_params']['beta'])
    
    # 调用所有绘图函数
    plot_fitting_function_comparison(results, OUTPUT_DIR, color_map)
    plot_fitting_function_frequency_comparison(results, OUTPUT_DIR, color_map)
    plot_all_std_dev_comparison(results, OUTPUT_DIR, color_map)
    
    print(f"\nAll summary plots have been saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()