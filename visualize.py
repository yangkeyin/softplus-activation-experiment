import pickle
import os
import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':
    with open('experiment_results/results.pkl', 'rb') as f:
        results = pickle.load(f)
    x_test = results['X_test']
    y_test = results['y_test']
    train_results = results['train_results']
    sort_indices = np.argsort(x_test.flatten())

    # 2. 根据索引对所有数组进行排序
    x_test = x_test[sort_indices]
    y_test = y_test[sort_indices]

    for epoch in train_results.keys():
        epoch_dir = os.path.join('experiment_results', f'epoch_{epoch}') # 读取epoch，为每个epoch生成一个目录
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)  
        # 绘制epoch下对应的每个neuron std图
        plt.figure(figsize=(10, 6))

        # 在该目录下，生成每个neuron的图
        for n in train_results[epoch].keys():
            plt.figure(figsize=(10, 6))
            plt.plot(x_test, y_test, label='sin(x)')
            plt.scatter(x_test, y_test, s=10, alpha=0.5)
            # 读取所有beta对应的y_pred和y_pred_stds
            for beta in train_results[epoch][n].keys():
                # 计算y_pred的平均值和y_pred_std的平均值
                y_preds = [train_results[epoch][n][beta][seed]['y_pred'] for seed in train_results[epoch][n][beta].keys()]
                y_preds = np.mean(y_preds, axis=0)[sort_indices]
                y_pred_std = np.mean([train_results[epoch][n][beta][seed]['y_pred_std'] for seed in train_results[epoch][n][beta].keys()])
                train_results[epoch][n][beta]['mean_std'] = y_pred_std
                # 绘制拟合图像
                plt.plot(x_test, y_preds, label=f'beta={beta}') 
                # 绘制之前的点
                plt.scatter(x_test, y_preds, s=10, alpha=0.5)
            plt.title(f'Neuron {n} - Epoch {epoch}')
            plt.xlabel('Sample Index')
            plt.ylabel('Predicted Value')
            plt.legend()
            plt.savefig(os.path.join(epoch_dir, f'neuron_{n}.png'))
            plt.close()

            betas = sorted(train_results[epoch][n].keys())
            mean_stds = [train_results[epoch][n][beta]['mean_std'] for beta in betas]
            # 打印x坐标轴
            plt.xticks(betas)
            plt.plot(betas, mean_stds, 'o-', label=f'n={n}') # 用 label 区分不同的 n
        

        plt.title(f'Mean Error Std Dev vs. Beta (Epoch: {epoch})')
        # plt.xscale('log')
        plt.xlabel('Beta (β)')
        plt.ylabel('Mean Standard Deviation of Error')
        
        plt.legend() # 显示 label
        plt.grid(True, which="both", linestyle='--')
        plt.savefig(os.path.join(epoch_dir, f'std_comparison_{epoch}.png'))
        plt.close()