# **实验提案 (PRP): 1D-CNN 频域偏见分析 (模仿 Figure 2a)**

## **1\. 实验目标**

本实验的核心目标是**复现 amplitude\_Fredformer.py 中 Figure 2(a) 的实验设计**，但将研究对象从 Transformer 替换为 1D-CNN。

我们将探究不同**卷积核大小 (Kernel Size)** 的 CNN，在面对不同频率振幅分布的数据集时，是否会表现出类似于 Transformer 的“频率偏见”（例如，偏好学习低频信号）。

我们将通过对比两个场景来验证这一点：

* **场景1 (低频偏置):** 训练一个 CNN，使其拟合一个“低频振幅强、高频振幅弱”的信号。  
* **场景2 (高频偏置):** 训练一个 CNN，使其拟合一个“低频振幅弱、高频振幅强”的信号。

## **2\. 核心逻辑 (规划 new\_scripts/run\_cnn\_freq\_bias.py 脚本)**

此 PRP 规划了新脚本 new\_scripts/run\_cnn\_freq\_bias.py 的实现细节。

### **A. 依赖 (Dependencies)**

* torch (nn, optim, utils.data)  
* numpy (及 numpy.fft)  
* matplotlib.pyplot  
* os  
* (不再需要 src.utils)

### **B. 模型定义 (Model Definition)**

我们将使用 Simple1DCNN 模型，kernel\_size 是其关键变量。

import torch.nn as nn  
import torch.nn.functional as F

class Simple1DCNN(nn.Module):  
    def \_\_init\_\_(self, kernel\_size):  
        super(Simple1DCNN, self).\_\_init\_\_()  
        \# 确保卷积后的长度不变  
        padding \= (kernel\_size \- 1\) // 2  
          
        self.conv\_stack \= nn.Sequential(  
            \# 输入: (Batch, 1, Length)  
            nn.Conv1d(in\_channels=1, out\_channels=16,   
                      kernel\_size=kernel\_size, padding=padding),  
            nn.ReLU(),  
            nn.Conv1d(in\_channels=16, out\_channels=32,   
                      kernel\_size=kernel\_size, padding=padding),  
            nn.ReLU()  
        )  
          
        \# 使用一个 1x1 卷积（等效于全连接）来聚合通道  
        self.output\_layer \= nn.Conv1d(in\_channels=32, out\_channels=1,   
                                      kernel\_size=1)  
        \# 输出: (Batch, 1, Length)

    def forward(self, x):  
        \# x 形状: (Batch, 1, Length)  
        x \= self.conv\_stack(x)  
        x \= self.output\_layer(x)  
        return x

### **C. 数据生成 (Data Generation)**

我们将创建两个独立的、模仿 Figure 2(a) 设计的数据集。

1. **信号配置:**  
   * N\_POINTS \= 200 (信号长度 L，对应 amplitude\_Fredformer 中的 SEQ\_LEN)  
   * **KEY\_FREQS\_K \= \[20, 40, 60\]** (k1, k2, k3 \- **关键频率分量 k**。选择这些值是为了在 N=200 上清晰可辨)  
   * NOISE\_LEVEL \= 0.1  
2. **振幅配置 (核心):**  
   * **AMPS\_SCENARIO\_1 (低频偏置): \[1.5, 1.0, 0.5\]** (k1振幅 \> k2振幅 \> k3振幅)  
   * **AMPS\_SCENARIO\_2 (高频偏置): \[0.5, 1.0, 1.5\]** (k1振幅 \< k2振幅 \< k3振幅)  
3. **数据生成函数 (generate\_data\_scenario):**  
   * **输入:** amps\_list, key\_freqs\_k\_list, n\_samples  
   * t \= torch.arange(N\_POINTS)  
   * **循环 n\_samples 次:**  
     * y\_signal \= torch.zeros(N\_POINTS)  
     * for amp, k\_freq in zip(amps\_list, key\_freqs\_k\_list):  
       * phase \= np.random.uniform(0, 2 \* np.pi)  
       * y\_signal \+= amp \* torch.sin(2 \* np.pi \* k\_freq \* t / N\_POINTS \+ phase)  
     * noise \= torch.randn(N\_POINTS) \* NOISE\_LEVEL  
     * Y.append(y\_signal.reshape(1, N\_POINTS))  
     * X.append((y\_signal \+ noise).reshape(1, N\_POINTS))  
   * **输出:** (X\_data, Y\_data) (形状 \[N, 1, N\_POINTS\])

### **D. 实验流程 (Experiment Workflow)**

1. **配置 (Config):**  
   * KERNEL\_SIZES\_TO\_TEST \= \[3, 25\] (对比 "高通" vs "低通" 两种极端情况)  
   * EPOCHS \= 2000  
   * EVAL\_STEP \= 50 (每 50 个 epoch 评估一次相对误差)  
   * LR \= 0.001  
   * N\_SAMPLES\_TRAIN \= 2000  
   * N\_SAMPLES\_TEST \= 400  
   * SCENARIOS \= {"Scenario\_1\_LowFreqBias": AMPS\_SCENARIO\_1, "Scenario\_2\_HighFreqBias": AMPS\_SCENARIO\_2}  
   * DEVICE \= torch.device(...)  
2. **初始化结果字典:** results \= {}  
3. **辅助函数 (使用 FFT):**  
   * def get\_avg\_spectrum(data\_tensor):  
     * data\_np \= data\_tensor.cpu().numpy().squeeze() (形状 \[N\_samples, N\_POINTS\])  
     * fft\_data \= np.fft.fft(data\_np, axis=1)  
     * fft\_mag \= np.abs(fft\_data)  
     * avg\_fft\_mag \= np.mean(fft\_mag, axis=0)  
     * return avg\_fft\_mag\[:N\_POINTS // 2\] (只返回正频率部分)  
   * def get\_avg\_relative\_error(pred\_tensor, target\_tensor, key\_indices\_k):  
     * pred\_fft\_mag \= np.abs(np.fft.fft(pred\_tensor.cpu().numpy().squeeze(), axis=1))  
     * target\_fft\_mag \= np.abs(np.fft.fft(target\_tensor.cpu().numpy().squeeze(), axis=1))  
     * errors \= \[\]  
     * for k in key\_indices\_k:  
       * true\_mag\_k \= np.mean(target\_fft\_mag\[:, k\]) (在批次上取平均)  
       * pred\_mag\_k \= np.mean(pred\_fft\_mag\[:, k\]) (在批次上取平均)  
       * if true\_mag\_k \< 1e-6: errors.append(0.0)  
       * else: errors.append(np.abs(pred\_mag\_k \- true\_mag\_k) / true\_mag\_k)  
     * return np.array(errors)  
4. **外层循环 (遍历场景):** for scenario\_name, amps in SCENARIOS.items():  
   * print(f"--- 运行场景: {scenario\_name} \---")  
   * results\[scenario\_name\] \= {}  
   * **生成数据:**  
     * (X\_train, Y\_train) \= generate\_data\_scenario(amps, KEY\_FREQS\_K, N\_SAMPLES\_TRAIN)  
     * (X\_test, Y\_test) \= generate\_data\_scenario(amps, KEY\_FREQS\_K, N\_SAMPLES\_TEST)  
   * **计算参考频谱 (使用 FFT):**  
     * avg\_target\_spectrum \= get\_avg\_spectrum(Y\_test.to(DEVICE))  
     * avg\_input\_spectrum \= get\_avg\_spectrum(X\_test.to(DEVICE))  
     * results\[scenario\_name\]\['avg\_target\_spectrum'\] \= avg\_target\_spectrum  
     * results\[scenario\_name\]\['avg\_input\_spectrum'\] \= avg\_input\_spectrum  
   * key\_indices\_k \= KEY\_FREQS\_K (关键索引现在就是 k 值)  
5. **内层循环 (遍历Kernel):** for kernel\_size in KERNEL\_SIZES\_TO\_TEST:  
   * print(f"--- 正在测试 Kernel Size \= {kernel\_size} \---")  
   * model \= Simple1DCNN(kernel\_size=kernel\_size).to(DEVICE)  
   * optimizer \= torch.optim.Adam(model.parameters(), lr=LR), criterion \= nn.MSELoss()  
   * error\_history \= \[\]  
   * **训练模型 (Training Loop):** for epoch in range(EPOCHS):  
     * model.train()  
     * ...(训练步骤)...  
     * **周期性评估 (Periodic Evaluation):**  
     * if (epoch \+ 1\) % EVAL\_STEP \== 0:  
       * model.eval()  
       * with torch.no\_grad(): Y\_pred \= model(X\_test)  
       * avg\_errors \= get\_avg\_relative\_error(Y\_pred, Y\_test, key\_indices\_k)  
       * error\_history.append(avg\_errors)  
       * print(...)  
   * **保存最终结果:**  
     * model.eval()  
     * with torch.no\_grad(): Y\_pred\_final \= model(X\_test)  
     * test\_mse \= criterion(Y\_pred\_final, Y\_test).item()  
     * avg\_pred\_spectrum\_final \= get\_avg\_spectrum(Y\_pred\_final)  
     * results\[scenario\_name\]\[kernel\_size\] \= {  
       * 'test\_mse': test\_mse,  
       * 'avg\_pred\_spectrum': avg\_pred\_spectrum\_final,  
       * 'error\_history': np.array(error\_history)  
     * }

## **3\. 关键指标 (Key Metrics)**

1. **test\_mse:** 最终拟合精度。  
2. **avg\_input\_spectrum:** 测试集输入 X\_test 的真实平均频谱 (FFT, \[0..N/2\])。  
3. **avg\_target\_spectrum:** 目标信号 Y\_test 的真实平均频谱 (FFT, \[0..N/2\])。  
4. **avg\_pred\_spectrum:** CNN 最终输出 Y\_pred 的平均频谱 (FFT, \[0..N/2\])。  
5. **error\_history:** 一个 2D 数组，形状为 (N\_eval\_steps, N\_key\_freqs)，记录了在训练过程中，模型在测试集上对每个关键频率的**平均相对误差**。

## **4\. 可视化方案 (Visualization Plan)**

**(已按您的要求更新为 FFT 版本)**

脚本将为**每个 kernel\_size** 生成一个单独的汇总图，该图**同时包含两个场景**，以模仿 amplitude\_Fredformer 的 Figure 2a 布局。

* **文件名示例:** figures/CNN\_freq\_bias/cnn\_bias\_k3\_comparison.png, figures/CNN\_freq\_bias/cnn\_bias\_k25\_comparison.png  
* **布局:** 2x2 多子图 (Subplots)。  
* **X轴 (频谱图):** k\_axis \= np.arange(N\_POINTS // 2\)

### **图 1: CNN 频域偏见分析 (Kernel Size \= 3\)**

* **文件名:** cnn\_bias\_k3\_comparison.png  
* **标题 (Fig):** "CNN 偏见分析 \- Kernel Size \= 3"  
* **gs \= fig.add\_gridspec(2, 2\)**  
* **子图 ax\[0, 0\] (左上): 场景 1 频谱 (低频偏置)**  
  * 标题: "场景 1 (低频偏置) 频谱"  
  * Y 轴: "Amplitude" (线性刻度)  
  * X 轴: "F (Frequency Component k)"  
  * 内容:  
    * ax\[0, 0\].plot(k\_axis, results\['Scenario\_1'\]\['avg\_input\_spectrum'\], 'b--', label='Input (Lookback)')  
    * ax\[0, 0\].plot(k\_axis, results\['Scenario\_1'\]\['avg\_target\_spectrum'\], 'g-', label='Ground Truth')  
    * ax\[0, 0\].plot(k\_axis, results\['Scenario\_1'\]\[3\]\['avg\_pred\_spectrum'\], 'r-', label='Forecasting (k=3)')  
  * 图例: "Input", "Ground Truth", "Forecasting"  
  * X轴限制: ax\[0, 0\].set\_xlim(0, max(KEY\_FREQS\_K) \* 1.5)  
* **子图 ax\[1, 0\] (左下): 场景 1 误差演化**  
  * 标题: "场景 1 相对误差"  
  * Y 轴: 关键频率 (例如 \[f'k={k}' for k in KEY\_FREQS\_K\])  
  * X 轴: 评估步骤 (Epoch Step)  
  * 内容:  
    * data \= results\['Scenario\_1'\]\[3\]\['error\_history'\].T  
    * im1 \= ax\[1, 0\].imshow(data, aspect='auto', cmap='gray\_r', vmin=0, vmax=1.0)  
* **子图 ax\[0, 1\] (右上): 场景 2 频谱 (高频偏置)**  
  * 标题: "场景 2 (高频偏置) 频谱"  
  * Y 轴: "Amplitude" (线性刻度)  
  * X 轴: "F (Frequency Component k)"  
  * 内容:  
    * ax\[0, 1\].plot(k\_axis, results\['Scenario\_2'\]\['avg\_input\_spectrum'\], 'b--', label='Input (Lookback)')  
    * ax\[0, 1\].plot(k\_axis, results\['Scenario\_2'\]\['avg\_target\_spectrum'\], 'g-', label='Ground Truth')  
    * ax\[0, 1\].plot(k\_axis, results\['Scenario\_2'\]\[3\]\['avg\_pred\_spectrum'\], 'r-', label='Forecasting (k=3)')  
  * X轴限制: ax\[0, 1\].set\_xlim(0, max(KEY\_FREQS\_K) \* 1.5)  
* **子图 ax\[1, 1\] (右下): 场景 2 误差演化**  
  * 标题: "场景 2 相对误差"  
  * Y 轴: 关键频率 (例如 \[f'k={k}' for k in KEY\_FREQS\_K\])  
  * X 轴: 评估步骤 (Epoch Step)  
  * 内容:  
    * data \= results\['Scenario\_2'\]\[3\]\['error\_history'\].T  
    * im2 \= ax\[1, 1\].imshow(data, aspect='auto', cmap='gray\_r', vmin=0, vmax=1.0)  
* **Colorbar:** 在图表右侧添加一个共享的 Colorbar (基于 im1 或 im2)。

### **图 2: CNN 频域偏见分析 (Kernel Size \= 25\)**

* **文件名:** cnn\_bias\_k25\_comparison.png  
* **(布局与图1完全相同, 只是所有数据都使用 kernel\_size=25 的结果)**  
  * 例如 results\['Scenario\_1'\]\[25\]\['avg\_pred\_spectrum'\]  
  * 以及 results\['Scenario\_1'\]\[25\]\['error\_history'\]

## **5\. 预期产物 (Deliverables)**

1. **Python 脚本:** new\_scripts/run\_cnn\_freq\_bias.py (根据本 PRP 规划创建)。  
2. **可视化图表 (共 2 个):**  
   * figures/CNN\_freq\_bias/cnn\_bias\_k3\_comparison.png (Kernel=3 的左右场景对比图)  
   * figures/CNN\_freq\_bias/cnn\_bias\_k25\_comparison.png (Kernel=25 的左右场景对比图)