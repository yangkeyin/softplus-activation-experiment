import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 输出目录配置 - 与 PRP 保持一致
FINETUNE_OUTPUT_DIR = "figures/beta_finetune_notinclude_1"
os.makedirs(FINETUNE_OUTPUT_DIR, exist_ok=True)
# 字体设置 - 解决中文显示和负号显示问题
# 使用Windows系统常见的中文字体，避免大量字体查找错误
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 确保负号显示正常



def load_results():
    """加载基线和微调结果"""
    baseline_file = os.path.join("figures/beta_base", "results_base.pkl")
    finetune_file = os.path.join(FINETUNE_OUTPUT_DIR, "results_finetune.pkl")
    
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


def generate_beta_local_fit_evolution(results_finetune, beta, scenario_name, beta_dir):
    """生成 Beta 内部演化总结图表 - 按照 PRP A.2 节要求"""
    print(f"生成 beta={beta}, 情景={scenario_name} 的局部拟合误差演化图表")
    
    # 获取所有种子
    seeds = list(results_finetune[beta].keys())
    
    # 获取所有记录的epoch
    all_epochs = []
    for seed in seeds:
        if scenario_name in results_finetune[beta][seed]:
            all_epochs.extend(list(results_finetune[beta][seed][scenario_name].keys()))
    all_epochs = sorted(list(set(all_epochs)))
    
    # 收集每个epoch的local_fit_error
    epoch_values = []
    mean_values = []
    std_values = []
    
    for epoch in all_epochs:
        mean_val, std_val = calculate_mean_std(results_finetune, beta, scenario_name, epoch, "local_fit_error")
        if mean_val is not None:
            epoch_values.append(epoch)
            mean_values.append(mean_val)
            std_values.append(std_val)
    
    if epoch_values:
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 绘制均值和标准差范围
        plt.semilogy(epoch_values, mean_values, '-o', color='blue', label="Mean local_fit_error")
        plt.fill_between(epoch_values, 
                        np.maximum(1e-10, np.array(mean_values) - np.array(std_values)),  # 确保log尺度下有效
                        np.array(mean_values) + np.array(std_values), 
                        color='blue', alpha=0.2, label="Standard Deviation")
        
        # 设置图表属性
        plt.title(f"Local Fit Error vs. Epoch (Beta={beta}, Scenario={scenario_name})")
        plt.xlabel("Finetune Epoch")
        plt.ylabel("Local Fit Error (log scale)")
        plt.grid(True)
        plt.legend()
        
        # 保存图表
        plot_file = os.path.join(beta_dir, "beta_local_fit_vs_epoch.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"保存局部拟合误差演化图表到 {plot_file}")


def generate_per_beta_plots(results_base, results_finetune, beta_values, scenarios, max_epoch):
    """为每个beta和情景生成深度分析图表（1x3布局）"""
    x_test = results_base['x_test']
    y_test = results_base['y_test']
    true_coef = results_base['true_coef']
    
    for beta in beta_values:
        for scenario_name in scenarios:
            print(f"生成 beta={beta}, 情景={scenario_name} 的深度分析图表")
            
            # 创建目录结构 - 按照 PRP 要求
            scenario_dir = os.path.join(FINETUNE_OUTPUT_DIR, scenario_name)
            beta_dir = os.path.join(scenario_dir, f"beta_{beta}")
            os.makedirs(scenario_dir, exist_ok=True)
            os.makedirs(beta_dir, exist_ok=True)
            
            # 获取所有记录的epoch
            seeds = list(results_finetune[beta].keys())
            all_epochs = []
            for seed in seeds:
                if scenario_name in results_finetune[beta][seed]:
                    all_epochs.extend(list(results_finetune[beta][seed][scenario_name].keys()))
            all_epochs = sorted(list(set(all_epochs)))
            
            # 为每个epoch生成图表
            for epoch in all_epochs:
                # 创建1x3网格图 - 按照 PRP 要求
                fig, axes = plt.subplots(1, 3, figsize=(24, 8))
                fig.suptitle(f"异常点微调深度分析 - Beta={beta}, 情景={scenario_name}, Epoch={epoch}", fontsize=16)
                
                # 获取均值和标准差
                y_pred_ft_mean, y_pred_ft_std = calculate_mean_std(results_finetune, beta, scenario_name, epoch, "y_pred_test_ft")
                y_pred_base_mean, _ = calculate_mean_std(results_finetune, beta, scenario_name, epoch, "y_pred_test_base")
                
                # 获取用于散点图的微调训练集数据（使用第一个种子的数据）
                x_train_ft_scatter = None
                y_train_ft_scatter = None
                for seed in seeds:
                    if scenario_name in results_finetune[beta][seed] and epoch in results_finetune[beta][seed][scenario_name]:
                        x_train_ft_scatter = results_finetune[beta][seed][scenario_name][epoch]['x_train_ft_scatter']
                        y_train_ft_scatter = results_finetune[beta][seed][scenario_name][epoch]['y_train_ft_scatter']
                        break
                
                if y_pred_ft_mean is not None and y_pred_base_mean is not None:
                    # 子图1: 拟合函数对比图 - 按照 PRP 要求
                    ax1 = axes[0]
                    # 绘制真实函数
                    ax1.plot(x_test.flatten(), y_test.flatten(), 'k-', label="True Function")
                    # 绘制基线预测
                    ax1.plot(x_test.flatten(), y_pred_base_mean, 'g-', label="Baseline Prediction")
                    # 绘制微调预测（主线和范围）
                    ax1.plot(x_test.flatten(), y_pred_ft_mean, 'r-', label="Finetuned Prediction")
                    ax1.fill_between(x_test.flatten(), y_pred_ft_mean - y_pred_ft_std, 
                                    y_pred_ft_mean + y_pred_ft_std, color='red', alpha=0.2)
                    # 添加测试点散点
                    ax1.scatter(x_test.flatten(), y_test.flatten(), color='gray', s=10, alpha=0.5, label="Test Points")
                    # 添加训练点散点（含异常点）
                    if x_train_ft_scatter is not None and y_train_ft_scatter is not None:
                        ax1.scatter(x_train_ft_scatter.flatten(), y_train_ft_scatter.flatten(), 
                                  color='blue', s=50, label="Train + Outlier Points")
                    
                    # 计算样本数量
                    N_train = len(results_base['x_train'])
                    N_outlier = len(scenarios[scenario_name])
                    N_test = len(x_test)
                    ax1.set_title(f"Fitted Function (Train: {N_train}+{N_outlier}, Test: {N_test})")
                    ax1.set_xlabel("x")
                    ax1.set_ylabel("y")
                    ax1.legend()
                    ax1.grid(True)
                    
                    # 子图2: 微调影响图 - 按照 PRP 要求
                    y_distortion_rms_mean, _ = calculate_mean_std(results_finetune, beta, scenario_name, epoch, "y_distortion_rms")
                    distortion_mean = y_pred_ft_mean - y_pred_base_mean
                    distortion_std = np.sqrt(y_pred_ft_std**2 + y_pred_ft_std**2)  # 误差传播
                    
                    ax2 = axes[1]
                    ax2.plot(x_test.flatten(), distortion_mean, 'b-', label="Distortion (Finetuned - Baseline)")
                    ax2.fill_between(x_test.flatten(), distortion_mean - distortion_std, 
                                    distortion_mean + distortion_std, color='blue', alpha=0.2)
                    ax2.scatter(x_test.flatten(), distortion_mean, color='blue', s=10, alpha=0.5)
                    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
                    ax2.set_title(f"Finetuning Impact (Avg. y_distortion_rms: {y_distortion_rms_mean:.4f})")
                    ax2.set_xlabel("x")
                    ax2.set_ylabel("Distortion")
                    ax2.legend()
                    ax2.grid(True)
                    
                    # 子图3: 频谱对比 - 按照 PRP 要求
                    spectrum_error_ft_mean, _ = calculate_mean_std(results_finetune, beta, scenario_name, epoch, "spectrum_error_ft")
                    pred_coef_ft_mean, pred_coef_ft_std = calculate_mean_std(results_finetune, beta, scenario_name, epoch, "pred_coef_ft")
                    pred_coef_base_mean, _ = calculate_mean_std(results_finetune, beta, scenario_name, epoch, "pred_coef_base")
                    
                    ax3 = axes[2]
                    k = np.arange(len(true_coef))
                    ax3.semilogy(k, np.abs(true_coef), 'k-', label="True Coef")
                    ax3.semilogy(k, np.abs(pred_coef_base_mean), 'g-', label="Baseline Coef")
                    ax3.semilogy(k, np.abs(pred_coef_ft_mean), 'r-', label="Finetuned Coef")
                    ax3.fill_between(k, np.abs(pred_coef_ft_mean) - np.abs(pred_coef_ft_std), 
                                    np.abs(pred_coef_ft_mean) + np.abs(pred_coef_ft_std), color='red', alpha=0.2)
                    ax3.scatter(k, np.abs(true_coef), color='black', s=30)
                    ax3.scatter(k, np.abs(pred_coef_base_mean), color='green', s=20)
                    ax3.scatter(k, np.abs(pred_coef_ft_mean), color='red', s=20)
                    ax3.set_title(f"Spectrum Comparison (Avg. spectrum_error_ft: {spectrum_error_ft_mean:.4f})")
                    ax3.set_xlabel("Frequency k")
                    ax3.set_ylabel("|Coefficient|")
                    ax3.legend()
                    ax3.grid(True)
                    
                    plt.tight_layout(rect=[0, 0, 1, 0.97])
                    plot_file = os.path.join(beta_dir, f"epoch_{epoch}.png")
                    plt.savefig(plot_file)
                    plt.close(fig)
                    print(f"保存图表到 {plot_file}")
            
            # 生成 Beta 内部演化总结图表 - 按照 PRP A.2 节要求
            generate_beta_local_fit_evolution(results_finetune, beta, scenario_name, beta_dir)


def generate_summary_plots(results_base, results_finetune, beta_values, scenarios, max_epoch):
    """生成全局汇总对比图表 - 按照 PRP 要求"""
    # 1. 按Epoch演化的指标图表 - 1x3布局
    for scenario_name in scenarios:
        print(f"生成情景 {scenario_name} 的指标演化图表")
        
        # 创建目录结构 - 按照 PRP 要求
        scenario_dir = os.path.join(FINETUNE_OUTPUT_DIR, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)
        
        # 获取所有记录的epoch
        beta = beta_values[0]
        seeds = list(results_finetune[beta].keys())
        epochs = list(results_finetune[beta][seeds[0]][scenario_name].keys())
        
        # 为每个beta值准备颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(beta_values)))
        
        # 创建1x3网格图 - 按照 PRP 要求（移除了 global_damage_rms）
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f"微调指标演化 - 情景={scenario_name}", fontsize=16)
        
        # 子图1: test_rms_ft vs finetune_epoch
        ax1 = axes[0]
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
        
        # 子图2: spectrum_error_ft vs finetune_epoch
        ax2 = axes[1]
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
        
        # 子图3: local_fit_error vs finetune_epoch
        ax3 = axes[2]
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
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        # 保存到情景目录下 - 按照 PRP 要求
        plot_file = os.path.join(scenario_dir, "summary_metrics_vs_epoch.png")
        plt.savefig(plot_file)
        plt.close(fig)
        print(f"保存图表到 {plot_file}")
    
    # 2. Beta权衡图表 - 作为额外分析
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
    
    # 保存到主目录下
    plot_file = os.path.join(FINETUNE_OUTPUT_DIR, "summary_tradeoff_vs_beta.png")
    plt.savefig(plot_file)
    plt.close(fig)
    print(f"保存图表到 {plot_file}")


def main():
    print("===== 异常点微调可视化开始 =====")
    print("加载结果文件...")
    results_base, results_finetune = load_results()
    
    if results_base is None or results_finetune is None:
        return
    
    # 获取配置参数
    BETA = results_base['BETA']
    # 从微调结果中获取异常点情景配置
    sample_beta = BETA[0]
    sample_seed = list(results_finetune[sample_beta].keys())[0]
    OUTLIER_SCENARIOS = {}
    
    # 动态获取情景配置
    for scenario_name in results_finetune[sample_beta][sample_seed]:
        # 尝试从第一个种子和第一个epoch获取异常点列表
        epochs = list(results_finetune[sample_beta][sample_seed][scenario_name].keys())
        if epochs:
            first_epoch = epochs[0]
            # 从名称推断异常点配置
            if 'Single' in scenario_name:
                OUTLIER_SCENARIOS[scenario_name] = [(0.0, 5.0)]
            elif 'Multiple' in scenario_name:
                OUTLIER_SCENARIOS[scenario_name] = [(0.0, 5.0), (np.pi/2, -3.0), (-np.pi, 2.0)]
    
    # 如果动态获取失败，使用默认配置
    if not OUTLIER_SCENARIOS:
        OUTLIER_SCENARIOS = {
            'Scenario_1_Single': [(0.0, 5.0)],
            'Scenario_2_Multiple': [(0.0, 5.0), (np.pi/2, -3.0), (-np.pi, 2.0)]
        }
    
    MAX_FINETUNE_EPOCHS = 1000
    
    print(f"检测到的异常点情景: {list(OUTLIER_SCENARIOS.keys())}")
    print("生成逐个Beta深度分析图表...")
    generate_per_beta_plots(results_base, results_finetune, BETA, OUTLIER_SCENARIOS, MAX_FINETUNE_EPOCHS)
    
    print("生成全局汇总对比图表...")
    generate_summary_plots(results_base, results_finetune, BETA, OUTLIER_SCENARIOS, MAX_FINETUNE_EPOCHS)
    
    print("\n所有可视化图表生成完成！")
    print(f"图表保存在: {FINETUNE_OUTPUT_DIR}")
    print(f"目录结构: {FINETUNE_OUTPUT_DIR}/[scenario_name]/[beta_*/]/epoch_*.png")
    print(f"局部拟合误差演化图表: {FINETUNE_OUTPUT_DIR}/[scenario_name]/[beta_*/]/beta_local_fit_vs_epoch.png")
    print(f"汇总图表: {FINETUNE_OUTPUT_DIR}/[scenario_name]/summary_metrics_vs_epoch.png")
    print(f"Beta权衡图表: {FINETUNE_OUTPUT_DIR}/summary_tradeoff_vs_beta.png")


if __name__ == "__main__":
    main()