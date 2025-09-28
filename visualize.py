import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, NullLocator

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_results")

if __name__ == '__main__':
    with open(f'{OUTPUT_DIR}/results.pkl', 'rb') as f:
        results = pickle.load(f)
    x_test = results['X_test']
    y_test = results['y_test']
    x_train = results['X_train']
    y_train = results['y_train']
    train_results = results['train_results']
    sort_indices = np.argsort(x_test.flatten())

    # 2. 根据索引对所有数组进行排序
    x_test = x_test[sort_indices]
    y_test = y_test[sort_indices]

    for epoch in train_results.keys():
        epoch_dir = os.path.join(OUTPUT_DIR, f'epoch_{epoch}') # 读取epoch，为每个epoch生成一个目录
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)  
        # 绘制epoch下对应的每个neuron std图
        plt.figure(figsize=(10, 6))
        ax = plt.gca() # 获取axes对象，方便更精细地调控
        # 使用all_mean_stds保存所有neuron的mean_std
        all_mean_stds = []

        # 在该目录下，生成每个neuron的图
        for n in train_results[epoch].keys():
            fig_fit, ax_fit = plt.subplots(figsize=(10, 6))
            ax_fit.plot(x_test, y_test, label='sin(x)')
            ax_fit.scatter(x_test, y_test, label='Test Data', color='green', s=10, alpha=0.5)
            ax_fit.scatter(x_train, y_train, label='Train Data', color='red', marker='x', s=10, alpha=0.4)

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
                ax_fit.plot(x_test, y_preds, label=f'beta={beta}') 
                # 绘制之前的点
                ax_fit.scatter(x_test, y_preds, s=10, alpha=0.5)

                # 绘制neuron-error图
                abs_error = np.abs(y_preds - y_test.flatten())
                ax_error.plot(x_test, abs_error, label=f'beta={beta}')

            ax_fit.set_title(f'Neuron {n} - Epoch {epoch}')
            ax_fit.set_xlabel('x')
            ax_fit.set_ylabel('Predicted Value')
            ax_fit.legend()
            fig_fit.savefig(os.path.join(epoch_dir, f'neuron_{n}.png'))
            plt.close(fig_fit)

            ax_error.set_title(f'Absolute Error for Neuron {n} - Epoch {epoch}')
            ax_error.set_xlabel('Test x')
            ax_error.set_ylabel('Absolute Error')
            ax_error.legend()
            fig_error.savefig(os.path.join(epoch_dir, f'neuron_{n}_error.png'))
            plt.close(fig_error)

            betas = sorted(train_results[epoch][n].keys())
            mean_stds = [train_results[epoch][n][beta]['mean_std'] for beta in betas]
            # 保存所有neuron的mean_std
            all_mean_stds.extend(mean_stds)
            ax.plot(betas, mean_stds, 'o-', label=f'n={n}') # 用 label 区分不同的 n
        

        ax.set_title(f'Mean Error Std Dev vs. Beta (Epoch: {epoch})')
        # plt.xscale('log')
        ax.set_xlabel('Beta (β)')
        ax.set_ylabel('Mean Standard Deviation of Error')
        
        ax.set_xscale('log')
        ax.set_yscale('log')

        # 强制设置x轴的刻度
        ax.set_xticks(betas)
        # 使用ScalFormatter设置x轴的刻度格式
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        # 强制设置y轴的刻度
        ax.set_yticks(sorted(list(set(all_mean_stds))))
        # 使用ScalFormatter设置y轴的刻度格式
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.yaxis.set_minor_locator(NullLocator())

        ax.legend() # 显示 label
        ax.grid(True, which="both", linestyle='--')
        plt.savefig(os.path.join(epoch_dir, f'std_comparison_{epoch}.png'))
        plt.close()