# **实验提案 (PRP): Transformer 频率学习极限检测**

## **1\. 实验目标**

本实验的核心目标是遵循用户的指导，探究 SimpleTransformerModel (源自 amplitude\_Fredformer.py) 是否能“依据频率条件学习到相应的信号”，并量化其“频率极限”。

我们将通过以下步骤实现这一目标：

1. **验证条件生成**：训练一个 Transformer 模型，使其学会根据输入的条件（横坐标x，频率 freq 和相位 phase）来生成对应的正弦波信号 y \= sin(freq \* x \+ phase)。  
2. **量化频率极限**：  
   * 在**训练过的频率**（插值）上评估模型的准确性 (MSE)。  
   * 在**未见过的频率**上评估，这里频率既包含训练频率内部（内插但未见过）也包含外部频率（外推）。观察模型性能（MSE）何时急剧下降，从而确定其“频率极限”。

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
   * SimpleTransformerModel 的 \_\_init\_\_ 函数将被修改以接受 input\_dim=3。这是为了处理 \[x, freq, phase\] 格式的条件输入。  
   * **输入形状**：模型将接受 (Batch, SeqLen, 3\) 的张量。SeqLen 可以是一个较小的值（例如 10），因为输入条件 \[x, freq, phase\] 在序列中是重复的。  
   * **输出形状**：模型将输出 (Batch, PredLen, 1\) 的张量。PredLen 是我们定义的信号点数（例如 100）。

### **C. 数据生成 (Data Generation)**

1. **generate\_data 函数**：  
   * **输入**：n\_samples (样本数), freqs\_list (用于采样的频率列表), seq\_len, pred\_len, x\_axis\_pred。  
   * **过程**：  
     1. 为 n\_samples 个样本，从 freqs\_list 中随机选择 freq，并随机生成 phase (范围 \[0, 2\*pi\])。  
     2. **创建 X (输入)**：生成一个 (n\_samples, seq\_len, 3\) 的张量。对于每个样本，seq\_len 维度上的所有 \[x, freq, phase\] 向量都是相同的。  
     3. **创建 Y (目标)**：生成一个 (n\_samples, pred\_len, 1\) 的张量。通过计算 y \= np.sin(freq \* x\_axis\_pred \+ phase) 得到目标信号。  
   * **输出**：返回一个 TensorDataset(X\_tensor, Y\_tensor)。

### **D. 实验流程 (Experiment Workflow)**

1. **配置 (Config)**：  
   * FREQS\_TRAIN \= \[1.0, 5.0, 10.0\] （一组用于训练的低中频率）  
   * FREQS\_TEST \= \[1.0, 3.0, 5.0, 8.0, 10.0, 12.0, 20.0\] （包含训练过的频率和未见过的频率）  
   * N\_SAMPLES\_TRAIN \= 5000  
   * N\_SAMPLES\_VAL \= 1000  
   * EPOCHS \= 50 （或根据收敛情况调整）  
   * SEEDS \= \[100, 200, 300, 400, 500\] （多个随机种子以进行多次实验，确保结果的鲁棒性）
2. **步骤 1: 训练 (Train)**  
   * **循环 SEEDS 中的每个种子**：  
     1. 设置随机种子。  
     2. 调用 generate\_data 创建训练集和验证集（均使用 FREQS\_TRAIN）。  
     3. 创建 DataLoader。  
     4. 实例化 SimpleTransformerModel(input\_dim=3, output\_dim=1, ...)。  
     5. 使用 Adam 优化器和 MSELoss 损失函数进行训练。  
     6. 在每轮 (epoch) 结束后，使用验证集评估模型，记录 train loss 和 val loss。  
     7. 在每个eval epoch后，**切换到评估模式**，对每个 FREQS\_TEST 中的 freq 生成测试数据，计算 mse 和 fits (Y\_test, Y\_pred)，记录每个 epoch 的结果。  
     8. 保存验证损失最低的“最佳模型” (best\_freq\_model\_seed{seed}.pth)。  
     9. 记录每个种子的 train loss, val loss, 和每个 epoch 的 mse 和 fits 历史，用于可视化。
3. **步骤 2: 总结 (Summarize)**  
   * 计算多个种子的平均 MSE 和 fits，用于最终可视化。  
   * **调用可视化函数**，传入所有记录的数据，以生成新的可视化图表。

## **3\. 关键指标 (Key Metrics)**

* **均方误差 (Mean Squared Error, MSE)**：MSE(Y\_pred, Y\_test)。这是量化模型在特定频率下拟合精度的核心指标。

## **4\. 可视化方案 (Visualization Plan)**

脚本将生成多个图表并保存到 figures/Transformer\_freq\_limit/ 目录：

### **A. 图 1: 训练损失图 (training_loss.png)**

* **类型**：线图 (Line Plot)。  
* **X 轴**：训练轮次 (Epochs)。  
* **Y 轴**：损失值 (Loss)，使用对数尺度。  
* **内容**：绘制多个种子的 train loss 和 val loss 曲线（平均以及fillbetween所有种子曲线）。  log图
* **目的**：展示训练过程中的收敛情况。

### **B. 图 2: 每个 epoch 的频率拟合图 (freq_fits_epoch{epoch}.png)**

* **类型**：多子图 (Subplots)，每个测试频率一个子图。  
* **X 轴**：时间/相位 (X\_AXIS\_PRED)。  
* **Y 轴**：信号幅度 (Amplitude)。  
* **内容**：每个子图显示该 epoch 下该频率的拟合情况：  
  1. 真实信号 (True Signal)：蓝色实线。  
  2. 预测信号 (Predicted Signal)：红色虚线。  
* **标题**：每个子图注明 Freq \= {freq} 和 MSE \= {mse:.6f}。  
* **目的**：展示每个 epoch 模型在所有测试频率上的拟合表现。每个 epoch 生成一个图。

### **C. 图 3: MSE vs Freq 随 epoch 变化图 (mse_vs_freq_over_epochs.png)**

* **类型**：线图 (Line Plot)。  
* **X 轴**：测试频率 (Test Frequency)。  
* **Y 轴**：均方误差 (MSE)，使用对数尺度。  
* **内容**：绘制不同 epoch 的 MSE vs Freq 曲线，每条曲线代表一个 epoch，使用渐变颜色表示 epoch 进展。  
* **特征**：标记训练过的频率 (Trained Freq) 和未见过的频率 (Unseen Freq)。  
* **目的**：展示模型的频率极限如何随训练进展而变化，观察 MSE 在高频时的下降趋势。

## **5\. 预期产物 (Deliverables)**

1. **Python 脚本**：new\_scripts/run\_transformer\_freq\_limit.py （根据本 PRP 规划创建）。  
2. **模型文件**：figures/Transformer\_freq\_limit/best\_freq\_model\_seed{seed}.pth （每个种子一个）。  
3. **可视化图表**：  
   * figures/Transformer\_freq\_limit/training_loss.png (训练损失图)  
   * figures/Transformer\_freq\_limit/freq_fits_epoch{epoch}.png (每个 epoch 的频率拟合图)  
   * figures/Transformer\_freq\_limit/mse_vs_freq_over_epochs.png (MSE vs Freq 随 epoch 变化图)
