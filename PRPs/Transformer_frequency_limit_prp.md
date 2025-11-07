# **实验提案 (PRP): Transformer 频率学习极限检测**

## **1\. 实验目标**

本实验的核心目标是遵循用户的指导，探究 SimpleTransformerModel (源自 amplitude\_Fredformer.py) 是否能“依据频率条件学习到相应的信号”，并量化其“频率极限”。

我们将通过以下步骤实现这一目标：

1. **验证条件生成**：训练一个 Transformer 模型，使其学会根据输入的条件（频率 freq 和相位 phase）来生成对应的正弦波信号 y \= sin(freq \* x \+ phase)。  
2. **量化频率极限**：  
   * 在**训练过的频率**（插值）上评估模型的准确性 (MSE)。  
   * 在**未见过的更高频率**（外推）上评估，观察模型性能（MSE）何时急剧下降，从而确定其“频率极限”。

## **2\. 核心逻辑 (规划 new\_scripts/run\_transformer\_freq\_limit.py 脚本)**

此 PRP 详细规划了新脚本 new\_scripts/run\_transformer\_freq\_limit.py 的实现细节。

### **A. 依赖 (Dependencies)**

* torch (包括 nn, optim, utils.data)  
* numpy  
* matplotlib.pyplot  
* os

### **B. 模型定义 (Model Definition)**

1. **复制模型**：我们将从 new\_scripts/amplitude\_Fredformer.py 中完整复制 PositionalEncoding 和 SimpleTransformerModel 类。  
2. **关键修改 (Adaptation)**：  
   * SimpleTransformerModel 的 \_\_init\_\_ 函数将被修改以接受 input\_dim=2。这是为了处理 \[freq, phase\] 格式的条件输入。  
   * **输入形状**：模型将接受 (Batch, SeqLen, 2\) 的张量。SeqLen 可以是一个较小的值（例如 10），因为输入条件 \[freq, phase\] 在序列中是重复的。  
   * **输出形状**：模型将输出 (Batch, PredLen, 1\) 的张量。PredLen 是我们定义的信号点数（例如 100）。

### **C. 数据生成 (Data Generation)**

1. **固定时间轴**：我们将定义一个全局的、固定的输出时间轴 X\_AXIS\_PRED \= np.linspace(0, 4\*np.pi, 100)。模型需要学习在此轴上生成信号。  
2. **generate\_data 函数**：  
   * **输入**：n\_samples (样本数), freqs\_list (用于采样的频率列表), seq\_len, pred\_len, x\_axis\_pred。  
   * **过程**：  
     1. 为 n\_samples 个样本，从 freqs\_list 中随机选择 freq，并随机生成 phase (范围 \[0, 2\*pi\])。  
     2. **创建 X (输入)**：生成一个 (n\_samples, seq\_len, 2\) 的张量。对于每个样本，seq\_len 维度上的所有 \[freq, phase\] 向量都是相同的。  
     3. **创建 Y (目标)**：生成一个 (n\_samples, pred\_len, 1\) 的张量。通过计算 y \= np.sin(freq \* x\_axis\_pred \+ phase) 得到目标信号。  
   * **输出**：返回一个 TensorDataset(X\_tensor, Y\_tensor)。

### **D. 实验流程 (Experiment Workflow)**

1. **配置 (Config)**：  
   * FREQS\_TRAIN \= \[1.0, 2.0, 3.0, 5.0, 8.0, 10.0\] （一组用于训练的低中频率）  
   * FREQS\_TEST \= \[1.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0\] （包含训练过的频率和未见过的更高频率）  
   * N\_SAMPLES\_TRAIN \= 5000  
   * N\_SAMPLES\_VAL \= 1000  
   * EPOCHS \= 50 （或根据收敛情况调整）  
2. **步骤 1: 训练 (Train)**  
   * 调用 generate\_data 创建训练集和验证集（均使用 FREQS\_TRAIN）。  
   * 创建 DataLoader。  
   * 实例化 SimpleTransformerModel(input\_dim=2, output\_dim=1, ...)。  
   * 使用 Adam 优化器和 MSELoss 损失函数进行训练。  
   * 在每轮 (epoch) 结束后，使用验证集评估模型，并保存验证损失 (val loss) 最低的“最佳模型” (best\_freq\_model.pth)。  
3. **步骤 2: 评估 (Evaluate)**  
   * 加载训练好的 best\_freq\_model.pth。  
   * 初始化一个空字典 results\_mse \= {} 来存储评估结果。  
   * **遍历 FREQS\_TEST 列表中的每一个 freq**：  
     1. 调用 generate\_data 生成一小批**特定**于此 freq 的测试数据（例如，使用固定的 phase \= 0 以便公平比较）。  
     2. 将测试数据传入模型，得到 Y\_pred。  
     3. 计算 mse \= MSELoss(Y\_pred, Y\_test)。  
     4. 将结果存入字典：results\_mse\[freq\] \= mse.item()。  
     5. **调用可视化函数（见 4.B）**，传入 Y\_test, Y\_pred, freq, mse，以绘制该频率下的拟合对比图。  
4. **步骤 3: 总结 (Summarize)**  
   * 在评估循环结束后，results\_mse 字典将包含所有测试频率的 MSE。  
   * **调用可视化函数（见 4.A）**，传入 results\_mse 和 FREQS\_TRAIN，以绘制最终的“频率-误差”总结图。

## **3\. 关键指标 (Key Metrics)**

* **均方误差 (Mean Squared Error, MSE)**：MSE(Y\_pred, Y\_test)。这是量化模型在特定频率下拟合精度的核心指标。

## **4\. 可视化方案 (Visualization Plan)**

脚本将生成两个图表并保存到 figures/Transformer\_freq\_limit/ 目录：

### **A. 图 1: 频率-误差总结图 (freq\_limit\_mse\_summary.png)**

* **类型**：条形图 (Bar Chart)。  
* **X 轴**：测试频率 (Test Frequency)，例如 \[1.0, 3.0, ..., 20.0\]。  
* **Y 轴**：均方误差 (MSE)，使用**对数尺度 (log scale)** 以更好地显示差异。  
* **特征**：条形将根据该频率是否在 FREQS\_TRAIN 中被着色：  
  * **绿色**：训练过的频率 (Trained Freq)。  
  * **红色**：未见过的频率 (Unseen Freq)。  
* **目的**：**宏观**展示模型的“频率极限”。预期红色条形（高频）的 MSE 会显著高于绿色条形。

### **B. 图 2: 拟合细节图 (freq\_limit\_fits.png)**

* **类型**：多子图 (Subplots)，每个测试频率一个子图。  
* **X 轴**：时间/相位 (X\_AXIS\_PRED)。  
* **Y 轴**：信号幅度 (Amplitude)。  
* **内容**：每个子图将同时绘制：  
  1. **真实信号 (True Signal)**：Y\_test （例如，蓝色实线）。  
  2. **预测信号 (Predicted Signal)**：Y\_pred （例如，红色虚线）。  
* **标题**：每个子图的标题将注明 Freq \= {freq} 和 MSE \= {mse:.6f}。  
* **目的**：**微观**展示模型在每个频率上的具体表现。预期在低频时两条线重合良好，在高频时（如 15.0, 20.0）Y\_pred 将严重偏离 Y\_test。

## **5\. 预期产物 (Deliverables)**

1. **Python 脚本**：new\_scripts/run\_transformer\_freq\_limit.py （根据本 PRP 规划创建）。  
2. **模型文件**：figures/Transformer\_freq\_limit/best\_freq\_model.pth。  
3. **可视化图表**：  
   * figures/Transformer\_freq\_limit/freq\_limit\_mse\_summary.png (总结图)  
   * figures/Transformer\_freq\_limit/freq\_limit\_fits.png (细节图)