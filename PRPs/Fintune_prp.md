# **实验提案 (PRP): 异常点微调的局部性与平滑度 (Beta) 分析**

## **1\. 实验目标**

本实验的核心目标是**量化并可视化**一个训练良好的神经网络（已学习到sin(x)先验）在多大程度上“破坏”了它的全局知识，以去拟合一个或多个局部的、强烈的异常点。

我们将对比不同平滑度（beta）的模型，以探究您的核心问题：模型是如何通过“局部扭曲”（高Beta）或“全局污染”（低Beta）来处理这种冲突的。

## **2\. 核心逻辑 (run\_outlier\_finetune.py)**

此脚本是一个独立的微调和评估脚本。

### **A. 依赖 (Prerequisites)**

此脚本**假设**以下文件已由 new\_scripts/run\_beta\_save.py（或您修改后的版本）准备就绪：

1. **基线结果文件**：./figures/beta\_base/results\_base.pkl  
   * 包含：x\_train, y\_train, x\_test, y\_test, true\_coef  
   * 包含：一个按 \[beta\]\[seed\]\[epoch\] 组织的字典，内含基线指标 test\_rms\_base, spectrum\_error\_base, y\_pred\_test\_base 和 y\_pred\_train\_base。  
2. **模型目录**：./figures/beta\_base/models/  
   * 包含：所有预训练好的模型，例如 model\_beta\_2.0\_seed\_100\_epoch\_10000.pth。

### **B. 微调配置 (Finetune Configuration)**

1. **异常点情景 (Outlier Scenarios)**  
   * OUTLIER\_SCENARIOS \= {  
   * 'Scenario\_1\_Single': \[(0.0, 5.0)\],  
   * 'Scenario\_2\_Multiple': \[(0.0, 5.0), (np.pi/2, \-3.0), (-np.pi, 2.0)\]  
   * }  
2. **微调参数**：  
   * MAX\_FINETUNE\_EPOCHS \= 1000  
   * FINETUNE\_LR \= 0.0001  
   * FINETUNE\_EVAL\_STEP \= 10 (每10步评估一次，以观察连续演化)  
3. **诊断区域**：  
   * **局部点 (Local Points)**：在评估时，将**动态**使用当前情景的 x 值 (例如 \[0.0, np.pi/2, \-np.pi\])。  
   * **远场区域 (Far-Field)**：x\_far \= all x\_test where abs(x) \> pi (保留核心诊断指标)

### **C. 执行流程**

1. **加载基线数据**：加载 results\_base.pkl。获取 BETA, SEEDS, BASELINE\_EPOCH，以及 x\_train, y\_train, x\_test, y\_test。  
2. **初始化微调结果字典**：finetune\_results \= {}  
3. **外层循环**：for beta in BETA: ... for seed in SEEDS: ...  
4. **中层循环**：for scenario\_name, outlier\_list in OUTLIER\_SCENARIOS.items():  
   * **准备微调数据**：x\_train\_ft \= torch.cat(...), y\_train\_ft \= torch.cat(...) (根据 outlier\_list 动态创建)。  
5. **内层循环（微调）**：  
   * 加载对应的预训练模型：model.load\_state\_dict(...)  
   * 创建**新**优化器：optimizer \= Adam(model.parameters(), lr=FINETUNE\_LR)  
   * for epoch in range(MAX\_FINETUNE\_EPOCHS):  
     * 在 (x\_train\_ft, y\_train\_ft) 上训练一步。  
     * **如果 (epoch+1) % FINETUNE\_EVAL\_STEP \== 0:**  
       * **执行评估 (见下节)**，并将所有指标存入 finetune\_results\[beta\]\[seed\]\[scenario\_name\]\[epoch+1\]。  
6. **保存结果**：将 finetune\_results 字典保存为 FINETUNE\_OUTPUT\_DIR / results\_finetune.pkl。

## **3\. 关键指标 (Metrics) \- 存储在 .pkl 中**

这是回答您问题的核心。我们将在微调模型的**评估阶段**（步骤 C.5）计算以下指标。

### **A. 基础指标 (用于绘图)**

1. **y\_pred\_test\_ft**: (感性指标) 在 x\_test 上的预测值 (Numpy 数组)。  
2. **y\_pred\_train\_ft**: (感性指标) 在 x\_train 上的预测值 (Numpy 数组)。  
3. **x\_train\_ft\_scatter / y\_train\_ft\_scatter**: 保存完整的微调训练集（含异常点），用于绘图时的散点。

### **B. 数值反映指标 (量化)**

1. **train\_rms\_ft**: (数值指标) RMSE(y\_pred\_train\_ft, y\_train)。  
2. **test\_rms\_ft**: (数值指标) RMSE(y\_pred\_test\_ft, y\_test)（在**整个** x\_test 上计算）。  
3. **y\_distortion\_rms**: (新数值指标) RMSE(y\_pred\_test\_ft, y\_pred\_test\_base)。**量化“微调影响图”的幅度。**  
4. **spectrum\_error\_ft**: (数值指标) RMSE(coef\_finetuned, true\_coef)。  
5. **local\_fit\_error**: (新诊断指标) **平均**局部拟合误差。  
   * **计算**：mean( abs(model\_finetuned(x\_i) \- y\_i) )，其中 (x\_i, y\_i) 来自当前 outlier\_list。  
   * **洞察**：衡量模型“屈服”于**所有**异常点的平均程度。

## **4\. 最终产物 (results\_finetune.pkl)**

results\_finetune.pkl 将包含一个按 \[beta\]\[seed\]\[scenario\_name\]\[finetune\_epoch\] 组织的字典，每个条目包含上述 3+6=9 个关键指标。

## **5\. 可视化方案 (新脚本 visualize\_finetune.py)**

此脚本将加载 results\_base.pkl 和 results\_finetune.pkl，并生成所有图表。  
核心：所有绘图将首先计算均值 (mean) 和标准差 (std) (跨 seed 维度)，并使用 mean 绘制主线，使用 plt.fill\_between 绘制颜色范围（置信区间）。  
**新输出目录结构**:

FINETUNE_OUTPUT_DIR/
├── results_finetune.pkl
├── summary_tradeoff_vs_beta.png  (图B.2 - 最终权衡图)
│
├── Scenario_1_Single/
│   ├── summary_metrics_vs_epoch.png (图B.1 - 按Epoch演化)
│   │
│   ├── beta_1.0/
│   │   ├── epoch_10.png           (图A.1)
│   │   ├── epoch_20.png           (图A.1)
│   │   ├── ...
│   │   └── beta_local_fit_vs_epoch.png  (图A.2 - 新增图表)
│   ├── beta_2.0/
│   │   └── ...
│   └── ...
│
└── Scenario_2_Multiple/
    ├── summary_metrics_vs_epoch.png (图B.1 - 按Epoch演化)
    │
    ├── beta_1.0/
    │   └── ...
    └── ...


### **A. 图表类型1：逐个 Beta 深度分析 (Per-Beta Deep Dives)**

**采纳建议**：此图表现在是 **1x3 布局**。

* **文件路径**: FINETUNE\_OUTPUT\_DIR / \[scenario\_name\] / \[beta\] / epoch\_{epoch}.png  
* **布局**: 一行三列 (1x3)  
* **图表**:  
1. **子图1: 拟合函数对比图**  
   * **标题**: Fitted Function (Train: {N\_train}+{N\_outlier}, Test: {N\_test})  
   * **线 (Mean)**: y\_test (黑), y\_pred\_base (绿), y\_pred\_finetuned (红)  
   * **范围 (Fill)**: y\_pred\_finetuned (浅红)  
   * **散点 (Scatter)**:  
     * x\_test vs y\_test (灰色, 小点)  
     * x\_train\_ft\_scatter vs y\_train\_ft\_scatter (蓝色, 大点) (含异常点)  
2. **子图2: 微调影响图 (Finetuning Impact)**  
   * **标题**: Finetuning Impact (Avg. y\_distortion\_rms: {value:.4f})  
   * **线 (Mean)**: y\_pred\_finetuned \- y\_pred\_base (蓝)  
   * **范围 (Fill)**: (浅蓝)  
   * **散点 (Scatter)**: x\_test vs y\_pred\_finetuned \- y\_pred\_base (蓝色, 小点)  
   * **洞察**: *微调这个动作本身造成了多大的形状改变？*  
3. **子图3: 频谱对比 (Spectrum)**  
   * **标题**: Spectrum Comparison (Avg. spectrum\_error\_ft: {value:.4f})  
   * **线 (Mean)**: true\_coef (黑), coef\_base (绿), coef\_finetuned (红) (均为log尺度)  
   * **散点 (Scatter)**: k vs coef\_... (用小点标出)

**A.2. Beta 内部演化总结 (新增图表)**

* **文件路径**: FINETUNE_OUTPUT_DIR / [scenario_name] / [beta] / beta_local_fit_vs_epoch.png  
* **布局**: 单个图表 (1x1)  
* **X 轴**: finetune_epoch (从 10 到 MAX_FINETUNE_EPOCHS)  
* **Y 轴**: local_fit_error (log 尺度)  
* **线 (Mean)**: local_fit_error 在所有 seed 上的平均值（仅针对当前 beta）。  
* **范围 (Fill)**: local_fit_error 在所有 seed 上的标准差范围（浅色）。  
* **标题**: Local Fit Error vs. Epoch (Beta={beta}, Scenario={scenario})  
* **洞察**: 此图表专门展示了这一个beta值的模型在面对异常点时，其局部拟合误差的演化过程。

### **B. 图表类型2：全局汇总对比 (Summary Plots)**

将生成汇总所有 beta 的 .png 文件。

1. **图 summary\_metrics\_vs\_epoch.png** (按Epoch演化)  
   * **采纳建议**：此图表现在是 **1x3 布局**，并**移除了 global\_damage\_rms**。  
   * **文件路径**: FINETUNE\_OUTPUT\_DIR / \[scenario\_name\] / summary\_metrics\_vs\_epoch.png  
   * **布局**: 一行三列 (1x3)  
   * **子图**:  
     * 子图1: test\_rms\_ft (Y轴) vs. finetune\_epoch (X轴)  
     * 子图2: spectrum\_error\_ft (Y轴) vs. finetune\_epoch (X轴)  
     * 子图3: local\_fit\_error (Y轴) vs. finetune\_epoch (X轴)  
   * **线条**: 每个子图包含**多条线**（带颜色范围），每条线代表一个 beta 值。  
   * **洞察**: *模型是如何（以及多快）被破坏/扭曲的？*