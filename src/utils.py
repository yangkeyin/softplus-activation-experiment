# Utilities for generating orthogonal polynomials.

import numpy as np
from numpy import pi, sqrt, sin, dot, sum, sign, arange
from numpy.linalg import inv, qr, cholesky

import numpy.polynomial.chebyshev  as cheb
import numpy.polynomial.legendre   as lege
import numpy.polynomial.polynomial as poly

# ---绘制目标函数---
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


def rescale(x, data_range):
    return 2 * (x - np.min(data_range)) / (np.max(data_range) - np.min(data_range)) - 1  # 归一化到 [-1, 1]

def get_orthpoly(n_deg, f_weighting, n_extra_point = 10, return_coef = True, representation='chebyshev', integral_method='legendre'):
    """
     Get orthogonal polynomials with respect to the weighting (f_weighting).
     The polynomials are represented by coefficients of Chebyshev polynomials.
     Evaluate it as: np.dot(cheb.chebvander(set_of_x, n_deg), repr_coef)
     See weighted_orthpoly1.py for validation of this algorithm.
    """
    n_sampling = n_deg + 1 + n_extra_point
    # Do the intergration by discrete sampling at good points.
    if integral_method == 'chebyshev':
        # Here we use (first kind) Chebyshev points.
        x_sampling = sin( pi * (-(n_sampling-1)/2 + np.arange(n_sampling))/n_sampling )
        # Put together the weighting and the weighting of the Gauss-Chebyshev quadrature.
        diag_weight = np.array([f_weighting(x)/cheb.chebweight(x) for x in x_sampling]) * (pi/n_sampling)
    elif integral_method == 'legendre':
        # Use Gauss-Legendre quadrature.
        x_sampling, int_weight = lege.leggauss(n_sampling)
        diag_weight = np.array([f_weighting(x)*w for x, w in zip(x_sampling, int_weight)])

    if representation == 'chebyshev':
        V = cheb.chebvander(x_sampling, n_deg)
    else:
        V = lege.legvander(x_sampling, n_deg)
    
    # Get basis from chol of the Gramian matrix.
#    inner_prod_matrix = dot(V.T, diag_weight[:, np.newaxis] * V)
#    repr_coef = inv(cholesky(inner_prod_matrix).T)
    # QR decomposition should give more accurate result.
    repr_coef = inv(qr(sqrt(diag_weight[:, np.newaxis]) * V, mode='r'))
    repr_coef = repr_coef * sign(sum(repr_coef,axis=0))
    
    if return_coef:
        return repr_coef
    else:
        if representation == 'chebyshev':
            polys = [cheb.Chebyshev(repr_coef[:,i]) for i in range(n_deg+1)]
        else:
            polys = [lege.Legendre (repr_coef[:,i]) for i in range(n_deg+1)]
        return polys

def orthpoly_coef(f, f_weighting, n_deg, **kwarg):
    if f_weighting == 'chebyshev':
        x_sample = sin( pi * (-n_deg/2 + arange(n_deg+1))/(n_deg+1) )
        V = cheb.chebvander(x_sample, n_deg)
    elif f_weighting == 'legendre':
        x_sample = lege.legroots(1*(arange(n_deg+2)==n_deg+1)) # 先生成一个n_deg+2个项的勒让德多项式多项式，最高项系数为1，并求解其根
        V = sqrt(arange(n_deg+1)+0.5) * lege.legvander(x_sample, n_deg)
    else:
        x_sample = lege.legroots(1*(arange(n_deg+2)==n_deg+1))
        basis_repr_coef = get_orthpoly(n_deg, f_weighting, 100, **kwarg)
        V = dot(cheb.chebvander(x_sample, n_deg), basis_repr_coef)

    if f == None:
        # Return Pseudo-Vandermonde matrix.
        return V

    y_sample = f(x_sample)
    coef = np.linalg.solve(V, y_sample)
    return coef

# vim: et sw=4 sts=4

# --- 勒让德多项式相关参数 ---
N_POLY_ORDER = 100 # 勒让德多项式的阶数
V = orthpoly_coef(None, 'legendre', N_POLY_ORDER)
#x_sample = sin( pi * (-n_poly_order/2 + arange(n_poly_order+1))/(n_poly_order+1) )
x_sample = lege.legroots(1*(np.arange(N_POLY_ORDER+2)==N_POLY_ORDER+1)) # 生成n_poly_order+2个项的勒让德多项式多项式，最高项系数为1，并求解其根
get_fq_coef = lambda f: np.linalg.solve(V, f(x_sample)) # 函数，解线性方程组V * c = f(x_sample)的系数，这组系数就是函数f的频谱，c[0]代表最平缓的成分，c[k]代表越来越高频，变化越来越剧烈的部分
