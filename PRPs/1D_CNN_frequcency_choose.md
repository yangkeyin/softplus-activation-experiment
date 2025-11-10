# **实验提案 (PRP): 1D-CNN 频域偏见分析 (序列预测 L=H 版)**

## **1\. 实验目标**

本实验的核心目标是探究不同卷积核大小 (Kernel Size) 的 1D-CNN，在**序列预测 (Forecasting)** 任务中，是否会表现出“频率偏见”。

此版本将设置**预测长度 (PRED\_LEN) \= 历史长度 (SEQ\_LEN)**，以确保输入、目标和预测值共享相同的频谱域，方便直接对比。

我们将对比CNN在学习和**外推**（Extrapolate）不同频率分量时的能力。此任务设置旨在与 Transformer (如 Fredformer) 的实验进行**公平的、同任务的**对比。

我们将通过对比两个场景来验证这一点：

* **场景1 (低频偏置):** 训练一个 CNN，使其在“低频振幅强、高频振幅弱”的信号上进行预测。  
* **场景2 (高频偏置):** 训练一个 CNN，使其在“低频振幅弱、高频振幅强”的信号上进行预测。

## **2\. 核心逻辑 (规划 run\_cnn\_freq\_bias.py 脚本)**

此 PRP 规划了 new\_scripts/run\_cnn\_freq\_bias.py 脚本（预测 L=H 版）的实现细节。

### **A. 依赖 (Dependencies)**

* torch (nn, optim, utils.data)  
* numpy (及 numpy.fft)  
* matplotlib.pyplot  
* os

### **B. 模型定义 (Model Definition)**

我们将使用一个 1D-CNN 模型，该模型经过修改以执行序列到序列的预测。

class Simple1DCNN(nn.Module):  
    def \_\_init\_\_(self, kernel\_size, seq\_len, pred\_len):  
        super(Simple1DCNN, self).\_\_init\_\_()  
        \# 确保卷积后的长度不变  
        padding \= (kernel\_size \- 1\) // 2  
          
        self.conv\_stack \= nn.Sequential(  
            \# 输入: (Batch, 1, SEQ\_LEN)  
            nn.Conv1d(in\_channels=1, out\_channels=16,   
                      kernel\_size=kernel\_size, padding=padding),  
            nn.ReLU(),  
            nn.Conv1d(in\_channels=16, out\_channels=32,   
                      kernel\_size=kernel\_size, padding=padding),  
            nn.ReLU()  
            \# 输出: (Batch, 32, SEQ\_LEN)  
        )  
          
        \# 输出层：将 (32, SEQ\_LEN) 的特征映射到 (1, PRED\_LEN) 的预测  
        self.output\_layer \= nn.Sequential(  
            nn.Conv1d(in\_channels=32, out\_channels=1, kernel\_size=1), \# 聚合通道  
            \# 输出: (Batch, 1, SEQ\_LEN)  
            nn.Flatten(),  
            \# 输出: (Batch, SEQ\_LEN)  
            nn.Linear(seq\_len, pred\_len), \# 全连接层 (seq\_len \== pred\_len)  
            \# 输出: (Batch, PRED\_LEN)  
            nn.Unsqueeze(1)  
            \# 最终输出: (Batch, 1, PRED\_LEN)  
        )  
        \# 注意: 另一种设计是当 L=H 时，直接使用 1x1 卷积作为输出  
        \# self.output\_layer \= nn.Conv1d(in\_channels=32, out\_channels=1, kernel\_size=1)  
        \# (当前设计使用 Linear，保持 PRP\_CNN\_Forecast\_Bias.md 中的结构)

    def forward(self, x):  
        \# x 形状: (Batch, 1, SEQ\_LEN)  
        x \= self.conv\_stack(x)  
        x \= self.output\_layer(x)  
        return x

### **C. 数据生成 (Data Generation)**

数据生成将采用“先生成长序列 \-\> 再分割 \-\> 最后创建滑动窗口”的模式。

**1\. 信号配置:**

* SEQ\_LEN \= 200 (历史序列长度 L)  
* PRED\_LEN \= 200 (**预测序列长度 H, H \= L**)  
* TOTAL\_SERIES\_LENGTH \= 5000 (用于生成窗口的原始序列总长)  
* KEY\_FREQS\_K \= \[20, 40, 60\] (关键频率分量 k，**基于 L=200 定义**)  
* NOISE\_LEVEL \= 0.1

**2\. 振幅配置 (核心):**

* AMPS\_SCENARIO\_1 (低频偏置): \[15, 10, 5\]  
* AMPS\_SCENARIO\_2 (高频偏置): \[5, 10, 15\]

**3\. 数据生成函数 (create\_datasets):**

1. **generate\_time\_series:** 生成一个总长为 TOTAL\_SERIES\_LENGTH 的单一 full\_series。  
   * *频率 f 通过 k / SEQ\_LEN 计算。*  
2. **分割:** 将 full\_series 按 70/10/20 分割为 data\_train, data\_val, data\_test。  
3. **create\_sliding\_windows:** **分别在** data\_train, data\_val, data\_test 内部创建 (X, Y) 窗口对 (X 长度 SEQ\_LEN=200, Y 长度 PRED\_LEN=200)。  
4. **DataLoader:** 将 (X\_train, Y\_train) 等转换为 TensorDataset 和 DataLoader 以进行批量训练。

### **D. 实验流程 (Experiment Workflow)**

**1\. 配置 (Config):**

* KERNEL\_SIZES\_TO\_TEST \= \[3, 25\]  
* EPOCHS \= 2000  
* EVAL\_STEP \= 50  
* LR \= 0.001  
* BATCH\_SIZE \= 64  
* SCENARIOS \= {"Scenario\_1\_LowFreqBias": ..., "Scenario\_2\_HighFreqBias": ...}

**2\. 频率索引 (简化):**

* FFT 分析将在 X\_test (Lookback), Y\_test (Ground Truth) 和 Y\_pred (Forecasting) 上进行。  
* 由于 SEQ\_LEN \== PRED\_LEN \== 200，所有频谱共享相同的 X 轴。  
* **key\_indices\_k \= \[20, 40, 60\]** (无需转换)

**3\. 辅助函数 (使用 FFT):**

* **get\_avg\_spectrum(data\_tensor):**  
  * 输入: \[N, 1, L=200\]  
  * 计算 np.fft.fft (axis=1)  
  * 返回 avg\_fft\_mag\[:L // 2\] (即 \[:100\])  
* **相对误差计算 (在训练循环中实现):**  
  * 在评估步骤中，获取全部 Y\_pred\_final\_tensor 和 Y\_test (形状 \[N\_test, 1, 200\])。  
  * 计算 pred\_fft\_mag 和 target\_fft\_mag。  
  * 循环遍历 key\_indices\_k (\[20, 40, 60\])：  
    * true\_mag\_k \= np.mean(target\_fft\_mag\[:, k\])  
    * pred\_mag\_k \= np.mean(pred\_fft\_mag\[:, k\])  
    * 计算相对误差 np.abs(pred\_mag\_k \- true\_mag\_k) / true\_mag\_k  
  * 将 \[error\_k1, error\_k2, error\_k3\] 存入 error\_history。

**4\. 循环 (Loops):**

* **外层循环 (遍历场景):** for scenario\_name, amps in SCENARIOS.items():  
  * 调用 create\_datasets(...) 获取 train\_loader, val\_loader, test\_loader, X\_test, Y\_test。 (假设 create\_datasets 也返回 X\_test)  
  * 计算 avg\_target\_spectrum \= get\_avg\_spectrum(Y\_test)。  
  * **(新增)** 计算 avg\_input\_spectrum \= get\_avg\_spectrum(X\_test)。  
* **内层循环 (遍历Kernel):** for kernel\_size in KERNEL\_SIZES\_TO\_TEST:  
  * 初始化 model \= Simple1DCNN(kernel\_size, SEQ\_LEN, PRED\_LEN)。  
  * **训练循环 (Training Loop):** ...  
  * **周期性评估 (Periodic Evaluation):** ...  
  * **保存最终结果:** ...

## **3\. 关键指标 (Key Metrics)**

* test\_mse: 最终预测 MSE。  
* **(新增)** avg\_input\_spectrum: **历史信号 X\_test** (长度 L=200) 的真实平均频谱。  
* avg\_target\_spectrum: **目标信号 Y\_test** (长度 H=200) 的真实平均频谱。  
* avg\_pred\_spectrum: **CNN 最终输出 Y\_pred** (长度 H=200) 的平均频谱。  
* error\_history: (N\_eval\_steps, 3)，记录模型在 Y\_pred 上对三个关键频率的相对误差。

## **4\. 可视化方案 (Visualization Plan)**

为每个 kernel\_size 生成一个 2x2 对比图 (文件名: cnn\_bias\_k{kernel\_size}\_prediction\_L200\_comparison.png)。

* **X轴 (频谱图):** k\_axis \= np.arange(SEQ\_LEN // 2\) (即 0 到 99\)  
* **X轴限制:** (0, max(KEY\_FREQS\_K) \* 1.5) (例如 0 到 90\)  
* 子图$$0, 0$$  
  (场景 1 频谱) /$$0, 1$$  
  (场景 2 频谱):  
  * **(修改)** plot(k\_axis, results\[...\]\['avg\_input\_spectrum'\], 'b--', label='Input (Lookback)')  
  * plot(k\_axis, results\[...\]\['avg\_target\_spectrum'\], 'g-', label='Ground Truth (Y\_test)')  
  * plot(k\_axis, results\[...\]\[k\_size\]\['avg\_pred\_spectrum'\], 'r-', label='Forecasting (Y\_pred, k=...)')  
  * X 轴标签: Frequency  
* 子图  
  (场景 1 误差)   
  (场景 2 误差):  
  * imshow(results\[...\]\[k\_size\]\['error\_history'\].T)  
  * Y 轴标签: \[f'k={KEY\_FREQS\_K\[0\]}', f'k={KEY\_FREQS\_K\[1\]}', f'k={KEY\_FREQS\_K\[2\]}'\] (使用原始 k 值)  
  * X 轴标签: 评估步骤 (Epoch Step)  
* **Colorbar:** 在图表右侧添加一个共享的 Colorbar。

## **5\. 预期产物 (Deliverables)**

1. **Python 脚本:** new\_scripts/run\_cnn\_freq\_bias.py (根据此 L=H 提案修改)。  
2. **可视化图表 (共 2 个):** (假设输出目录更新)  
   * figures/CNN\_freq\_bias\_FORECAST\_L200/cnn\_bias\_k3\_prediction\_comparison.png  
   * figures/CNN\_freq\_bias\_FORECAST\_L200/cnn\_bias\_k25\_prediction\_comparison.png