from math import pi
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, NullLocator

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_results_beta_1")

# --- 重构思路：将两种分析模式封装到独立的函数中 ---

def run_beta_analysis(results):
    """
    执行标准的、以Beta为横坐标的分析和绘图。
    """
    print("Running in BETA analysis mode...")
    
    x_test = results['X_test']
    y_test = results['y_test']
    x_train = results['X_train']
    y_train = results['y_train']
    train_results = results['train_results']
    sort_indices = np.argsort(x_test.flatten())

    # 对所有数组进行排序
    x_test = x_test[sort_indices]
    y_test = y_test[sort_indices]

    for epoch in train_results.keys():
        epoch_dir = os.path.join(OUTPUT_DIR, f'epoch_{epoch}')
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        
        # 绘制epoch下对应的每个neuron std图
        fig_std, ax_std = plt.subplots(figsize=(10, 6))
        all_mean_stds = []

        # 在该目录下，生成每个neuron的图
        for n in train_results[epoch].keys():
            fig_fit, ax_fit = plt.subplots(figsize=(10, 6))
            ax_fit.plot(x_test, y_test, label='sin(x)')
            ax_fit.scatter(x_test, y_test, label='Test Data', color='green', s=10, alpha=0.5)
            ax_fit.scatter(x_train, y_train, label='Train Data', color='black', marker='x', s=10, alpha=0.4)

            # 绘制neuron-error图
            fig_error, ax_error = plt.subplots(figsize=(10, 6))

            # 读取所有beta对应的y_pred和y_pred_stds
            for beta in train_results[epoch][n].keys():
                # 计算y_pred的平均值和y_pred_std的平均值
                y_preds = [train_results[epoch][n][beta][seed]['y_pred'] for seed in train_results[epoch][n][beta].keys()]
                y_preds = np.mean(y_preds, axis=0)[sort_indices]
                y_pred_std = np.mean([train_results[epoch][n][beta][seed]['y_pred_std'] for seed in train_results[epoch][n][beta].keys()])
                train_results[epoch][n][beta]['mean_std'] = y_pred_std
                
                # 绘制拟合图像
                line = ax_fit.plot(x_test, y_preds)[0]
                # 设置标签
                line.set_label(f'beta={beta:.2f}')
                # 绘制之前的点
                ax_fit.scatter(x_test, y_preds, s=10, alpha=0.5, color=line.get_color())

                # 绘制neuron-error图
                error = y_preds - y_test.flatten()
                ax_error.plot(x_test, error, color=line.get_color(), label=line.get_label())

            ax_fit.set_title(f'Neuron {n} - Epoch {epoch}')
            ax_fit.set_xlabel('x')
            ax_fit.set_ylabel('Predicted Value')
            ax_fit.legend()
            fig_fit.savefig(os.path.join(epoch_dir, f'neuron_{n}.png'))
            plt.close(fig_fit)

            ax_error.set_title(f'Error for Neuron {n} - Epoch {epoch}')
            ax_error.set_xlabel('Test x')
            ax_error.set_ylabel('Error')
            ax_error.legend()
            fig_error.savefig(os.path.join(epoch_dir, f'neuron_{n}_error.png'))
            plt.close(fig_error)

            # 收集数据用于std比较图
            betas = sorted(train_results[epoch][n].keys())
            mean_stds = [train_results[epoch][n][beta]['mean_std'] for beta in betas]
            all_mean_stds.extend(mean_stds)
            ax_std.plot(betas, mean_stds, 'o-', label=f'n={n}')
            
        # 绘制std比较图
        ax_std.set_title(f'Mean Error Std Dev vs. Beta (Epoch: {epoch})')
        ax_std.set_xlabel('Beta (β)')
        ax_std.set_ylabel('Mean Standard Deviation of Error')
        ax_std.set_xscale('log')
        ax_std.set_yscale('log')
        
        # 设置x轴刻度
        ax_std.set_xticks(betas)
        ax_std.xaxis.set_major_formatter(plt.ScalarFormatter())
        
        # 设置y轴刻度
        ax_std.set_yticks(sorted(list(set(all_mean_stds))))
        ax_std.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax_std.yaxis.set_minor_locator(NullLocator())
        
        ax_std.legend()
        ax_std.grid(True, which="both", linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, f'std_vs_beta_comparison_{epoch}.png'))
        plt.close(fig_std)

def run_width_analysis(results):
    """
    执行以光滑宽度W为横坐标的分析和绘图。
    """
    print("Running in WIDTH analysis mode...")
    
    width_values = results['width_list']
    pi_labels = [f'{w/np.pi:.2f}π' for w in width_values]
    
    x_test = results['X_test']
    y_test = results['y_test']
    x_train = results['X_train']
    y_train = results['y_train']
    train_results = results['train_results']
    sort_indices = np.argsort(x_test.flatten())

    # 对所有数组进行排序
    x_test = x_test[sort_indices]
    y_test = y_test[sort_indices]

    for epoch in train_results.keys():
        epoch_dir = os.path.join(OUTPUT_DIR, f'epoch_{epoch}')
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        
        # 绘制epoch下对应的每个neuron std图
        fig_std, ax_std = plt.subplots(figsize=(10, 6))
        all_mean_stds = []

        # 在该目录下，生成每个neuron的图
        for n in train_results[epoch].keys():
            fig_fit, ax_fit = plt.subplots(figsize=(10, 6))
            ax_fit.plot(x_test, y_test, label='sin(x)')
            ax_fit.scatter(x_test, y_test, label='Test Data', color='green', s=10, alpha=0.5)
            ax_fit.scatter(x_train, y_train, label='Train Data', color='black', marker='x', s=10, alpha=0.4)

            # 绘制neuron-error图
            fig_error, ax_error = plt.subplots(figsize=(10, 6))

            # 读取所有beta对应的y_pred和y_pred_stds
            for beta in train_results[epoch][n].keys():
                # 计算当前beta对应的width
                beta_index = list(train_results[epoch][n].keys()).index(beta)
                pi_label = pi_labels[beta_index]

                # 计算y_pred的平均值和y_pred_std的平均值
                y_preds = [train_results[epoch][n][beta][seed]['y_pred'] for seed in train_results[epoch][n][beta].keys()]
                y_preds = np.mean(y_preds, axis=0)[sort_indices]
                y_pred_std = np.mean([train_results[epoch][n][beta][seed]['y_pred_std'] for seed in train_results[epoch][n][beta].keys()])
                train_results[epoch][n][beta]['mean_std'] = y_pred_std
                
                # 绘制拟合图像
                line = ax_fit.plot(x_test, y_preds)[0]
                # 设置标签
                line.set_label(f'width={pi_label}, beta={beta:.2f}')
                # 绘制之前的点
                ax_fit.scatter(x_test, y_preds, s=10, alpha=0.5, color=line.get_color())

                # 绘制neuron-error图
                error = y_preds - y_test.flatten()
                ax_error.plot(x_test, error, color=line.get_color(), label=line.get_label())

            ax_fit.set_title(f'Neuron {n} - Epoch {epoch}')
            ax_fit.set_xlabel('x')
            ax_fit.set_ylabel('Predicted Value')
            ax_fit.legend()
            fig_fit.savefig(os.path.join(epoch_dir, f'neuron_{n}.png'))
            plt.close(fig_fit)

            ax_error.set_title(f'Error for Neuron {n} - Epoch {epoch}')
            ax_error.set_xlabel('Test x')
            ax_error.set_ylabel('Error')
            ax_error.legend()
            fig_error.savefig(os.path.join(epoch_dir, f'neuron_{n}_error.png'))
            plt.close(fig_error)

            # 收集数据用于std比较图
            betas = train_results[epoch][n].keys()
            mean_stds = [train_results[epoch][n][beta]['mean_std'] for beta in betas]
            all_mean_stds.extend(mean_stds)
            ax_std.plot(width_values, mean_stds, 'o-', label=f'n={n}')
            
        # 绘制std比较图
        ax_std.set_title(f'Mean Error Std Dev vs. Smoothness Width (Epoch: {epoch})')
        ax_std.set_xlabel('Smoothness Width (W)')
        ax_std.set_ylabel('Mean Standard Deviation of Error')
        ax_std.set_yscale('log')

        # 使用两步法来设置X轴标签
        ax_std.set_xticks(width_values)
        ax_std.set_xticklabels(pi_labels, rotation=45, ha='right')
        
        # 设置y轴刻度
        ax_std.set_yticks(sorted(list(set(all_mean_stds))))
        ax_std.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax_std.yaxis.set_minor_locator(NullLocator())
        
        ax_std.legend()
        ax_std.grid(True, which="both", linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, f'std_vs_width_comparison_{epoch}.png'))
        plt.close(fig_std)


if __name__ == '__main__':
    # --- 主逻辑：加载数据，并根据数据内容决定调用哪个分析函数 ---
    
    with open(f'{OUTPUT_DIR}/results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    # 核心判断：如果结果文件中包含 'width_list'，则进入宽度分析模式
    if 'width_list' in results and results['width_list'] is not None:
        # 执行宽度分析模式
        run_width_analysis(results)
    else:
        # 否则，执行beta分析模式
        run_beta_analysis(results)