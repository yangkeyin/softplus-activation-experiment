# Investigating the Impact of Softplus Activation Smoothness
(探究Softplus激活函数的平滑度影响)

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A computer experiment to find the optimal smoothness/sharpness parameter (β) for the Softplus activation function when approximating the `sin(x)` function with a simple neural network.

这是一个计算机实验，旨在通过拟合`sin(x)`函数，找到Softplus激活函数最优的平滑度/锐度参数 (β)。

## 🎯 目标 (Objective)

根据通用逼近定理，具有非线性激活函数的神经网络可以拟合任意连续函数。本项目探究了一个更细致的问题：**激活函数的“形状”或“平滑度”如何影响拟合的质量？**

我们通过使用一个参数化的Softplus函数，系统性地调整其锐度（从非常平滑到接近ReLU），并量化其对拟合`sin(x)`任务的影响。

## 🔬 方法 (Methodology)

1.  **模型 (Model)**: 一个简单的全连接神经网络 (MLP)，包含一个可变数量神经元 (`n`) 的隐藏层。
2.  **激活函数 (Activation Function)**: 参数化的Softplus函数：
    $$ \text{Softplus}(x, \beta) = \frac{1}{\beta} \log(1 + e^{\beta x}) $$
    其中 `β` 控制函数的锐度。`β` 越大，函数越接近 `ReLU`。
3.  **任务 (Task)**: 在 `[-2π, 2π]` 区间上拟合 `y = sin(x)`。
4.  **参数扫描 (Parameter Sweep)**: 对不同的隐藏层神经元数量 `n` 和Softplus的 `β` 参数进行网格搜索。
5.  **评估指标 (Metric)**: 使用测试集上预测误差的标准差 `std(y_err)` 作为核心评估指标，它衡量了模型预测误差的一致性。

## 🚀 如何使用 (How to Use)

1.  **克隆仓库 (Clone the repository):**
    ```bash
    git clone [https://github.com/your-username/softplus-activation-experiment.git](https://github.com/your-username/softplus-activation-experiment.git)
    cd softplus-activation-experiment
    ```

2.  **创建虚拟环境并安装依赖 (Create a virtual environment and install dependencies):**
    建议创建一个`requirements.txt`文件来管理依赖。
    ```bash
    # (Optional) Create requirements.txt
    pip freeze > requirements.txt 
    
    # Install dependencies
    pip install -r requirements.txt 
    # Or manually: pip install torch numpy matplotlib scikit-learn
    ```

3.  **运行实验 (Run the experiment):**
    脚本 `run_experiment.py` 的所有参数都在文件顶部进行配置，您可以轻松调整。
    ```bash
    python run_experiment.py
    ```
    实验分为两个阶段：
    * **粗调 (Coarse Search)**: 默认运行一个大范围的 `β` 列表。
    * **精调 (Fine-grained Search)**: 分析第一次的结果图，在代码中注释掉粗调列表，并为你认为最优的 `β` 区间设置一个更精细的列表，然后再次运行。

4.  **查看结果 (Check the results):**
    所有输出，包括每次运行的损失曲线和最终的总结图，都会保存在 `experiment_results/` 目录下。

## 📈 结果与发现 (Results & Findings)

实验的核心输出是误差标准差 (`std(y_err)`) 随 `β` 变化的曲线图。

*(在这里插入您生成的最终总结图。当您运行完实验后，将 `summary_std_vs_beta.png` 拖拽到这个文件夹，然后使用下面的格式引用它)*

![Summary Plot](experiment_results/summary_std_vs_beta.png)

从上图可以看出：
* **过小或过大的β值效果不佳**: 当`β`非常小时，Softplus接近线性函数，模型表达能力不足。当`β`非常大时，Softplus接近ReLU，虽然表达能力强，但可能导致训练不稳定或梯度问题，误差反而增大。
* **存在一个“最优区间”**: 对于不同的神经元数量`n`，都存在一个相似的中间区域，使得`std(y_err)`达到最小值。这表明，对于特定任务，激活函数既不能过于平滑也不能过于尖锐。
* **神经元数量的影响**: 增加神经元数量 `n` 通常会降低整体的误差水平，但最优`β`区间的趋势是相似的。

## 💡 结论 (Conclusion)

本次实验验证了激活函数的形状对神经网络函数拟合能力有显著影响。对于`sin(x)`这个平滑的周期函数，一个中等锐度（例如 `β` 在 [最佳范围] 之间）的Softplus激活函数表现最好。这为在实际应用中选择或设计激活函数提供了有价值的参考。

## 📜 许可证 (License)

This project is licensed under the MIT License.
