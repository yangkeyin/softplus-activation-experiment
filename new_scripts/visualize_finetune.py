import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 输出目录配置
OUTPUT_DIR = "figures/beta_finetune/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_results():
    """加载基线和微调结果"""
    baseline_file = "figures/beta_base/results_base.pkl"
    finetune_file = "figures/beta_finetune/results_finetune.pkl"
    
    if not os.path.exists(baseline_file):
        print(f"错误：找不到基线结果文件 {baseline_file}")
        return None, None
    
    if not os.path.exists(finetune_file):
        print(f"错误：找不到微调结果文件 {finetune_file}")
        return None, None
    
    with open(baseline_file, 'rb') as f:
        results_base = pickle.load(f)
    
    with open(finetune_file, 'rb') as f:
        results_finetune = pickle.load(f)
    
    return results_base, results_finetune


def calculate_mean_std(results, beta, scenario_name, epoch, metric_key):
    """计算跨种子的均值和标准差"""
    values = []
    for seed in results[beta]:
        if scenario_name in results[beta][seed] and epoch in results[beta][seed][scenario_name]:
            if metric_key in results[beta][seed][scenario_name][epoch]:
                values.append(results[beta][seed][scenario_name][epoch][metric_key])
    
    if not values:
        return None, None
    
    mean_val = np.mean(values, axis=0)
    std_val = np.std(values, axis=0)
    return mean_val, std_val


def generate_per_beta_plots(results_base, results_finetune, beta_values, scenarios, max_epoch):
    """为每个beta和情景生成深度分析图表"""
    x_test = results_base['x_test']
    y_test = results_base['y_test']
    true_coef = results_base['true_coef']
    
    for beta in beta_values:
        for scenario_name in scenarios:
            print(f"生成 beta={beta}, 情景={scenario_name} 的深度分析图表")
            
            # 创建2x2网格图
            fig, axes = plt.subplots(2, 2, figsize=(20, 15))
            fig.suptitle(f"异常点微调深度分析 - Beta={beta}, 情景={scenario_name}", fontsize=16)
            
            # 获取最终epoch的结果
            # 子图1: 拟合函数对比图
            # 获取均值和标准差
            y_pred_ft_mean, y_pred_ft_std = calculate_mean_std(results_finetune, beta, scenario_name, max_epoch, "y_pred_test_ft")
            y_pred_base_mean, _ = calculate_mean_std(results_finetune, beta, scenario_name, max_epoch, "y_pred_test_base")
            
            # 获取用于散点图的微调训练集数据（使用第一个种子的数据）
            seeds = list(results_finetune[beta].keys())
            x_train_ft_scatter = None
            y_train_ft_scatter = None
            for seed in seeds:
                if scenario_name in results_finetune[beta][seed] and max_epoch in results_finetune[beta][seed][scenario_name]:
                    x_train_ft_scatter = results_finetune[beta][seed][scenario_name][max_epoch]['x_train_ft_scatter']
                    y_train_ft_scatter = results_finetune[beta][seed][scenario_name][max_epoch]['y_train_ft_scatter']
                    break
            
            if y_pred_ft_mean is not None and y_pred_base_mean is not None:
                ax1 = axes[0, 0]
                # 绘制真实函数
                ax1.plot(x_test.flatten(), y_test.flatten(), 'k-', label="True Function")
                # 绘制基线预测
                ax1.plot(x_test.flatten(), y_pred_base_mean, 'g-', label="Baseline Prediction")
                # 绘制微调预测（主线和范围）
                ax1.plot(x_test.flatten(), y_pred_ft_mean, 'r-', label="Finetuned Prediction")
                ax1.fill_between(x_test.flatten(), y_pred_ft_mean - y_pred_ft_std, 
                                y_pred_ft_mean + y_pred_ft_std, color='red', alpha=0.2)
                # 添加散点
                ax1.scatter(x_test.flatten(), y_test.flatten(), color='gray', s=10, alpha=0.5, label="Test Points")
                if x_train_ft_scatter is not None and y_train_ft_scatter is not None:
                    ax1.scatter(x_train_ft_scatter.flatten(), y_train_ft_scatter.flatten(), 
                              color='blue', s=50, label="Train + Outlier Points")
                
                N_train = len(results_base['x_train'])
                # 正确处理x_train_ft_scatter数据格式 - 它是一个二维数组，每个元素是单个值
                x_scatter_data = results_finetune[beta][seeds[0]][scenario_name][max_epoch]['x_train_ft_scatter']
                # 计算异常点数量 - 比较x值与异常点场景中的x坐标
                N_outlier = 0
                for x_val in x_scatter_data:
                    if any(abs(x_val[0] - o[0]) < 1e-6 for o in scenarios[scenario_name]):
                        N_outlier += 1
                N_test = len(x_test)
                ax1.set_title(f"Fitted Function (Train: {N_train}+{N_outlier}, Test: {N_test})")
                ax1.set_xlabel("x")
                ax1.set_ylabel("y")
                ax1.legend()
                ax1.grid(True)
                
                # 子图2: 微调误差图
                test_rms_ft_mean, _ = calculate_mean_std(results_finetune, beta, scenario_name, max_epoch, "test_rms_ft")
                error_mean = y_pred_ft_mean - y_test.flatten()
                error_std = y_pred_ft_std
                
                ax2 = axes[0, 1]
                ax2.plot(x_test.flatten(), error_mean, 'r-', label="Error (Finetuned - True)")
                ax2.fill_between(x_test.flatten(), error_mean - error_std, 
                                error_mean + error_std, color='red', alpha=0.2)
                ax2.scatter(x_test.flatten(), error_mean, color='red', s=10, alpha=0.5)
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
                ax2.set_title(f"Finetuned Error (Avg. test_rms_ft: {test_rms_ft_mean:.4f})")
                ax2.set_xlabel("x")
                ax2.set_ylabel("Error")
                ax2.legend()
                ax2.grid(True)
                
                # 子图3: 微调影响图
                y_distortion_rms_mean, _ = calculate_mean_std(results_finetune, beta, scenario_name, max_epoch, "y_distortion_rms")
                distortion_mean = y_pred_ft_mean - y_pred_base_mean
                distortion_std = np.sqrt(y_pred_ft_std**2 + y_pred_ft_std**2)  # 误差传播
                
                ax3 = axes[1, 0]
                ax3.plot(x_test.flatten(), distortion_mean, 'b-', label="Distortion (Finetuned - Baseline)")
                ax3.fill_between(x_test.flatten(), distortion_mean - distortion_std, 
                                distortion_mean + distortion_std, color='blue', alpha=0.2)
                ax3.scatter(x_test.flatten(), distortion_mean, color='blue', s=10, alpha=0.5)
                ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
                ax3.set_title(f"Finetuning Impact (Avg. y_distortion_rms: {y_distortion_rms_mean:.4f})")
                ax3.set_xlabel("x")
                ax3.set_ylabel("Distortion")
                ax3.legend()
                ax3.grid(True)
                
                # 子图4: 频谱对比
                spectrum_error_ft_mean, _ = calculate_mean_std(results_finetune, beta, scenario_name, max_epoch, "spectrum_error_ft")
                pred_coef_ft_mean, pred_coef_ft_std = calculate_mean_std(results_finetune, beta, scenario_name, max_epoch, "pred_coef_ft")
                pred_coef_base_mean, _ = calculate_mean_std(results_finetune, beta, scenario_name, max_epoch, "pred_coef_base")
                
                ax4 = axes[1, 1]
                k = np.arange(len(true_coef))
                ax4.semilogy(k, np.abs(true_coef), 'k-', label="True Coef")
                ax4.semilogy(k, np.abs(pred_coef_base_mean), 'g-', label="Baseline Coef")
                ax4.semilogy(k, np.abs(pred_coef_ft_mean), 'r-', label="Finetuned Coef")
                ax4.fill_between(k, np.abs(pred_coef_ft_mean) - np.abs(pred_coef_ft_std), 
                                np.abs(pred_coef_ft_mean) + np.abs(pred_coef_ft_std), color='red', alpha=0.2)
                ax4.scatter(k, np.abs(true_coef), color='black', s=30)
                ax4.scatter(k, np.abs(pred_coef_base_mean), color='green', s=20)
                ax4.scatter(k, np.abs(pred_coef_ft_mean), color='red', s=20)
                ax4.set_title(f"Spectrum Comparison (Avg. spectrum_error_ft: {spectrum_error_ft_mean:.4f})")
                ax4.set_xlabel("Frequency k")
                ax4.set_ylabel("|Coefficient|")
                ax4.legend()
                ax4.grid(True)
                
                plt.tight_layout(rect=[0, 0, 1, 0.97])
                plot_file = os.path.join(OUTPUT_DIR, f"beta_{beta}_scenario_{scenario_name}.png")
                plt.savefig(plot_file)
                plt.close(fig)
                print(f"保存图表到 {plot_file}")


def generate_summary_plots(results_base, results_finetune, beta_values, scenarios, max_epoch):
    """生成全局汇总对比图表"""
    # 1. 按Epoch演化的指标图表
    for scenario_name in scenarios:
        print(f"生成情景 {scenario_name} 的指标演化图表")
        
        # 创建2x2网格图
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(f"微调指标演化 - 情景={scenario_name}", fontsize=16)
        
        # 获取所有记录的epoch
        beta = beta_values[0]
        seeds = list(results_finetune[beta].keys())
        epochs = list(results_finetune[beta][seeds[0]][scenario_name].keys())
        
        # 为每个beta值准备颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(beta_values)))
        
        # 子图1: test_rms_ft
        ax1 = axes[0, 0]
        for i, beta in enumerate(beta_values):
            rms_values = []
            rms_std = []
            for epoch in epochs:
                mean_val, std_val = calculate_mean_std(results_finetune, beta, scenario_name, epoch, "test_rms_ft")
                if mean_val is not None:
                    rms_values.append(mean_val)
                    rms_std.append(std_val)
            if rms_values:
                rms_values = np.array(rms_values)
                rms_std = np.array(rms_std)
                ax1.semilogy(epochs, rms_values, '-o', label=f"Beta={beta}", color=colors[i])
                ax1.fill_between(epochs, rms_values - rms_std, rms_values + rms_std, color=colors[i], alpha=0.2)
        ax1.set_title("Test RMS vs Finetune Epoch")
        ax1.set_xlabel("Finetune Epoch")
        ax1.set_ylabel("Test RMS")
        ax1.legend()
        ax1.grid(True)
        
        # 子图2: spectrum_error_ft
        ax2 = axes[0, 1]
        for i, beta in enumerate(beta_values):
            spectrum_values = []
            spectrum_std = []
            for epoch in epochs:
                mean_val, std_val = calculate_mean_std(results_finetune, beta, scenario_name, epoch, "spectrum_error_ft")
                if mean_val is not None:
                    spectrum_values.append(mean_val)
                    spectrum_std.append(std_val)
            if spectrum_values:
                spectrum_values = np.array(spectrum_values)
                spectrum_std = np.array(spectrum_std)
                ax2.semilogy(epochs, spectrum_values, '-o', label=f"Beta={beta}", color=colors[i])
                ax2.fill_between(epochs, spectrum_values - spectrum_std, spectrum_values + spectrum_std, color=colors[i], alpha=0.2)
        ax2.set_title("Spectrum Error vs Finetune Epoch")
        ax2.set_xlabel("Finetune Epoch")
        ax2.set_ylabel("Spectrum Error")
        ax2.legend()
        ax2.grid(True)
        
        # 子图3: local_fit_error
        ax3 = axes[1, 0]
        for i, beta in enumerate(beta_values):
            local_error_values = []
            local_error_std = []
            for epoch in epochs:
                mean_val, std_val = calculate_mean_std(results_finetune, beta, scenario_name, epoch, "local_fit_error")
                if mean_val is not None:
                    local_error_values.append(mean_val)
                    local_error_std.append(std_val)
            if local_error_values:
                local_error_values = np.array(local_error_values)
                local_error_std = np.array(local_error_std)
                ax3.plot(epochs, local_error_values, '-o', label=f"Beta={beta}", color=colors[i])
                ax3.fill_between(epochs, local_error_values - local_error_std, local_error_values + local_error_std, color=colors[i], alpha=0.2)
        ax3.set_title("Local Fit Error vs Finetune Epoch")
        ax3.set_xlabel("Finetune Epoch")
        ax3.set_ylabel("Local Fit Error")
        ax3.legend()
        ax3.grid(True)
        
        # 子图4: global_damage_rms
        ax4 = axes[1, 1]
        for i, beta in enumerate(beta_values):
            global_damage_values = []
            global_damage_std = []
            for epoch in epochs:
                mean_val, std_val = calculate_mean_std(results_finetune, beta, scenario_name, epoch, "global_damage_rms")
                if mean_val is not None:
                    global_damage_values.append(mean_val)
                    global_damage_std.append(std_val)
            if global_damage_values:
                global_damage_values = np.array(global_damage_values)
                global_damage_std = np.array(global_damage_std)
                ax4.semilogy(epochs, global_damage_values, '-o', label=f"Beta={beta}", color=colors[i])
                ax4.fill_between(epochs, global_damage_values - global_damage_std, global_damage_values + global_damage_std, color=colors[i], alpha=0.2)
        ax4.set_title("Global Damage RMS vs Finetune Epoch")
        ax4.set_xlabel("Finetune Epoch")
        ax4.set_ylabel("Global Damage RMS")
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plot_file = os.path.join(OUTPUT_DIR, f"summary_metrics_vs_finetune_epoch_{scenario_name}.png")
        plt.savefig(plot_file)
        plt.close(fig)
        print(f"保存图表到 {plot_file}")
    
    # 2. Beta权衡图表
    print("生成Beta权衡图表")
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
    linestyles = ['-', '--']
    
    for i, (scenario_name, _) in enumerate(scenarios.items()):
        global_damage_values = []
        global_damage_std = []
        local_error_values = []
        local_error_std = []
        
        for beta in beta_values:
            # 获取最大epoch的结果
            mean_gd, std_gd = calculate_mean_std(results_finetune, beta, scenario_name, max_epoch, "global_damage_rms")
            mean_le, std_le = calculate_mean_std(results_finetune, beta, scenario_name, max_epoch, "local_fit_error")
            
            if mean_gd is not None and mean_le is not None:
                global_damage_values.append(mean_gd)
                global_damage_std.append(std_gd)
                local_error_values.append(mean_le)
                local_error_std.append(std_le)
        
        if global_damage_values and local_error_values:
            beta_array = np.array(beta_values[:len(global_damage_values)])
            global_damage_array = np.array(global_damage_values)
            global_damage_std_array = np.array(global_damage_std)
            local_error_array = np.array(local_error_values)
            local_error_std_array = np.array(local_error_std)
            
            # 绘制全局损伤（左轴）
            ax1.semilogy(beta_array, global_damage_array, '-o', label=f"{scenario_name} - Global Damage", 
                        color=colors[i], linestyle=linestyles[i])
            ax1.fill_between(beta_array, global_damage_array - global_damage_std_array, 
                            global_damage_array + global_damage_std_array, color=colors[i], alpha=0.2)
            
            # 绘制局部误差（右轴）
            ax2.plot(beta_array, local_error_array, '-s', label=f"{scenario_name} - Local Error", 
                    color=colors[i], linestyle=linestyles[i])
            ax2.fill_between(beta_array, local_error_array - local_error_std_array, 
                            local_error_array + local_error_std_array, color=colors[i], alpha=0.2)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Beta (log scale)')
    ax1.set_ylabel('Global Damage RMS (log scale)', color='red')
    ax2.set_ylabel('Local Fit Error', color='blue')
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title(f"Global Damage vs Local Error Tradeoff at Epoch {max_epoch}")
    plt.grid(True)
    plt.tight_layout()
    
    plot_file = os.path.join(OUTPUT_DIR, "summary_tradeoff_vs_beta.png")
    plt.savefig(plot_file)
    plt.close(fig)
    print(f"保存图表到 {plot_file}")


def main():
    print("加载结果文件...")
    results_base, results_finetune = load_results()
    
    if results_base is None or results_finetune is None:
        return
    
    # 获取配置参数
    BETA = results_base['BETA']
    OUTLIER_SCENARIOS = {
        'Scenario_1_Single': [(0.0, 5.0)],
        'Scenario_2_Multiple': [(0.0, 5.0), (np.pi/2, -3.0), (-np.pi, 2.0)]
    }
    MAX_FINETUNE_EPOCHS = 1000
    
    print("生成逐个Beta深度分析图表...")
    generate_per_beta_plots(results_base, results_finetune, BETA, OUTLIER_SCENARIOS, MAX_FINETUNE_EPOCHS)
    
    print("生成全局汇总对比图表...")
    generate_summary_plots(results_base, results_finetune, BETA, OUTLIER_SCENARIOS, MAX_FINETUNE_EPOCHS)
    
    print("\n所有可视化图表生成完成！")
    print(f"图表保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()