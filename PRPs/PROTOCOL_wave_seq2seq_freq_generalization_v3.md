# 实验协议: Transformer 频率泛化能力测试 (Frequency Generalization Test) v3

## 目录

1. [实验目标](#1-实验目标)
2. [核心配置](#2-核心配置)
3. [数据处理流程](#3-数据处理流程)
4. [训练与评估逻辑](#4-训练与评估逻辑)
5. [可视化与输出](#5-可视化与输出)
6. [运行指南](#6-运行指南)
7. [预期结果](#7-预期结果)

---

## 1. 实验目标

验证 **Transformer 模型**在学习了特定频率（如 10Hz, 20Hz）的正弦波模式后，是否具备以下能力：

### 1.1 插值泛化 (Interpolation)

- **定义**: 能否自动预测**未见过的中间频率**（如 15Hz, 25Hz）
- **意义**: 测试模型是否真正理解了频率的连续变化规律
- **难度**: 中等（频率在训练范围内）

### 1.2 外推泛化 (Extrapolation)

- **定义**: 能否自动预测**范围外的频率**（如 5Hz, 30Hz）
- **意义**: 测试模型是否能推广到超出训练范围的新频率
- **难度**: 高（最具挑战性）

---

## 2. 核心配置

### 2.1 频率设置

```python
FREQS_TRAIN = [10.0, 20.0]  # Hz
FREQS_TEST = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]  # Hz
```

| 频率 | 类型 | 说明 |
|------|------|------|
| 5.0 | 外推 | 低于最小训练频率 |
| 10.0 | 训练 | 参与模型参数更新 |
| 15.0 | 插值 | 在训练范围内但未见过 |
| 20.0 | 训练 | 参与模型参数更新 |
| 25.0 | 插值 | 在训练范围内但未见过 |
| 30.0 | 外推 | 高于最大训练频率 |

**关键约束**: 模型参数更新（Backpropagation）仅基于 10Hz 和 20Hz 的数据。

### 2.2 序列参数

```python
SEQ_LEN = 100              # 输入序列长度 (Lookback window)
PRED_LEN = 100             # 预测序列长度 (Forecasting horizon)
TOTAL_POINTS = 5000        # 全长时间点数
```

| 参数 | 值 | 说明 |
|------|-----|------|
| SEQ_LEN | 100 | 输入窗口，模型基于前100个点进行预测 |
| PRED_LEN | 100 | 输出窗口，模型预测后100个点 |
| TOTAL_POINTS | 5000 | 每个频率生成5000个采样点 |

**采样说明**:
- 采样范围: [0, 2π]，均匀采样 5000 个点
- 波形公式: $y = \sin(f \times x)$，其中 $x \in [0, 2\pi]$，$f$ 为频率倍数
- 10Hz 含义: 在 [0, 2π] 范围内完成 10 个完整周期
- 20Hz 含义: 在 [0, 2π] 范围内完成 20 个完整周期

### 2.3 数据集划分 (Data Splitting)

```python
RANGE_TRAIN_START = 0
RANGE_TRAIN_END = 3500
RANGE_TEST_START = 3500
RANGE_TEST_END = 5000
```

**数据硬切分** (Time-Series Hard Split):

```
时间轴 ├─────────────────────────┼─────────────┤
      0        训练范围(3500)    3500  测试范围(5000)

训练范围 (0-3500):
  - 点数: 3500 个采样点
  - 用途: 生成训练样本
  - 约束: 仅用于更新模型参数

测试范围 (3500-5000):
  - 点数: 1500 个采样点
  - 用途: 评估模型泛化能力
  - 约束: 完全不参与训练
```

**关键特性**:
- ✅ **时间上的严格分离**: 训练和测试数据来自完全不同的时间段
- ✅ **同频率同切分**: 所有频率使用相同的切分点（0-3500 vs 3500-5000）
- ✅ **无数据泄露**: 测试范围的数据不会在任何形式下进入训练

### 2.4 可视化控制

```python
VIS_EPOCHS = [1, 5, 10, 20, 30, 40, 50]      # 可视化的epoch列表
VIS_SAMPLE_INDICES = [0, 10, 20]              # 可视化的样本索引
```

**说明**:
- 仅在 `VIS_EPOCHS` 中的 epoch 生成详细可视化
- 不是每个 epoch 都保存图片（节省时间和空间）
- 每个可视化中展示固定的样本（通过 `VIS_SAMPLE_INDICES` 控制）

---

## 3. 数据处理流程

### 3.1 第一步: 生成全量数据

对于 `FREQS_TEST` 中的每一个频率，生成完整的 5000 点正弦波序列：

```python
def generate_wave_full(freq, sampling_rate, total_points):
    t = np.arange(total_points) / sampling_rate
    wave = np.sin(2 * np.pi * freq * t)
    return wave  # 形状: (5000,)
```

**输出**: 6 个频率 × 1 个波形 = 6 个数组，每个 5000 点

### 3.2 第二步: 双重切片 (Dual Slicing)

对于每一个频率，生成两组数据集：

```
频率 f:
  ├─ Set A (Train Range): 来自 0~3500 点
  └─ Set B (Test Range): 来自 3500~5000 点
```

#### Set A 转换为样本

从 `0-3500` 范围中，使用滑动窗口创建样本对：

```python
# 对于频率 f，范围 0-3500：
输入序列 (Input):  [点0, 点1, ..., 点99] (100 个点)
目标序列 (Target): [点100, 点101, ..., 点199] (100 个点)

输入序列 (Input):  [点1, 点2, ..., 点100]
目标序列 (Target): [点101, 点102, ..., 点200]

... (继续滑动)

输入序列 (Input):  [点3400, 点3401, ..., 点3499]
目标序列 (Target): [点3500, 点3501, ..., 点3599] ❌ 超出范围！
```

**样本数量**:
- Train Set 有效起始位置: 0 到 3500 - 100 - 100 = 3300
- 样本数量: 3301 个

#### Set B 转换为样本

类似逻辑，从 `3500-5000` 范围创建：

```python
# 对于频率 f，范围 3500-5000：
输入序列 (Input):  [点3500, 点3501, ..., 点3599]
目标序列 (Target): [点3600, 点3601, ..., 点3699]

... (继续滑动)

输入序列 (Input):  [点4800, 点4801, ..., 点4899]
目标序列 (Target): [点4900, 点4901, ..., 点4999] ✓ 恰好结束
```

**样本数量**:
- Test Set 有效起始位置: 3500 到 5000 - 100 - 100 = 4800
- 样本数量: 1301 个

### 3.3 第三步: 训练加载器组装

```python
# 仅使用训练频率的 Train Set
for freq in [10.0, 20.0]:
    samples_10hz_train = Convert(waves[10.0][0:3500])    # 3301 个样本
    samples_20hz_train = Convert(waves[20.0][0:3500])    # 3301 个样本

# 混合并打乱
train_loader = DataLoader(
    CombinedDataset([samples_10hz_train, samples_20hz_train]),
    batch_size=32,
    shuffle=True
)
```

**特性**:
- ✅ 仅包含 10Hz 和 20Hz
- ✅ 仅包含训练范围 (0-3500)
- ✅ 样本被打乱（Shuffled）以避免频率顺序偏差

---

## 4. 训练与评估逻辑

### 4.1 训练循环 (Training Loop)

```python
for epoch in range(1, EPOCHS + 1):
    # 使用混合的 training_loader 更新模型参数
    for batch in training_loader:
        src, tgt = batch
        pred = model(src, tgt)
        loss = MSELoss(pred, tgt)
        loss.backward()
        optimizer.step()
```

**关键点**:
- 只有 10Hz 和 20Hz 的数据参与梯度更新
- 优化器：Adam，学习率 0.001
- 调度器：每 20 个 epoch 降低 50%

### 4.2 评估循环 (Evaluation Loop)

在每个 `VIS_EPOCHS` 中的 epoch 后执行：

```python
if epoch in VIS_EPOCHS:
    for range_name in ['train', 'test']:
        for freq in FREQS_TEST:  # 包括所有6个频率
            # 冻结模型
            model.eval()
            
            # 获取该频率、该范围的所有样本
            dataset = WaveSeq2SeqDataset(
                wave=waves[freq],
                range_start=RANGE_TRAIN_START if range_name == 'train' else RANGE_TEST_START,
                range_end=RANGE_TRAIN_END if range_name == 'train' else RANGE_TEST_END
            )
            
            # 逐个样本进行预测
            for sample in dataset:
                pred = model(sample.input)
                mse = MSE(pred, sample.target)
                # 记录 mse, pred, truth
```

**评估矩阵**:

| 范围 | 频率 | 样本数 | MSE统计 |
|------|------|--------|--------|
| Train (0-3500) | 5.0Hz | 3301 | 均值±标准差 |
| Train (0-3500) | 10.0Hz | 3301 | 均值±标准差 |
| ... | ... | ... | ... |
| Test (3500-5000) | 5.0Hz | 1301 | 均值±标准差 |
| Test (3500-5000) | 10.0Hz | 1301 | 均值±标准差 |
| ... | ... | ... | ... |

**统计收集**:

对于每个 (范围, 频率, epoch) 三元组，收集一个 MSE 列表：

```python
MSE_list[频率][epoch] = [mse_sample_1, mse_sample_2, ..., mse_sample_N]
```

计算统计量：

```python
mean = np.mean(MSE_list[频率][epoch])
std = np.std(MSE_list[频率][epoch])
```

---

## 5. 可视化与输出

### 5.1 文件夹结构

```
figures/
└── Freq_Generalization_Exp_YYYYMMDD_HHMMSS/
    ├── config.json
    ├── train/
    │   ├── mse_vs_epoch_train_range.png
    │   ├── 5.0Hz/
    │   │   ├── epoch_001_samples.png
    │   │   ├── epoch_005_samples.png
    │   │   └── ...
    │   ├── 10.0Hz/
    │   ├── 15.0Hz/
    │   ├── 20.0Hz/
    │   ├── 25.0Hz/
    │   └── 30.0Hz/
    │
    └── test/
        ├── mse_vs_epoch_test_range.png
        ├── 5.0Hz/
        │   └── ... (结构同上)
        └── ... (其他频率)
```

### 5.2 代表性样本可视化: 子图矩阵设计

**文件**: `epoch_{X:03d}_samples.png`

**布局**: N 行 × 2 列，每行对应一个样本

#### 第一列: 拟合图 (Fitting)

展示时间序列的拟合情况：

```
Amplitude
    │     ╱╲    ╱╲
    │    ╱  ╲  ╱  ╲
    ├───┼────┼──────┼───────
    │  ╱      ╲╱      ╲
    │ ╱
    └─────────────────────── Time Steps
  Input  │  Ground Truth (Blue) + Prediction (Red)
         SEQ_LEN
```

**元素**:
- **黑色实线 (Input)**: 输入序列 (步骤 0-99)
- **蓝色实线 (Ground Truth)**: 真实目标序列 (步骤 100-199)
- **红色虚线 (Prediction)**: 模型预测序列 (步骤 100-199)
- **灰色虚线**: 分界线 (SEQ_LEN = 100)

**目的**: 直观判断预测是否准确、是否有相位偏移或幅度偏差

#### 第二列: 误差图 (Error / Residual)

展示预测部分的残差：

```
Error
    │     │   │
    │    ││  ││
    ├────┼┼──┼┼──────
    │   ││  ││
    │    │   │
    └─────────────────── Time Steps
         (步骤 100-199)
```

**计算**: 
$$\text{Residual}_t = \text{Prediction}_t - \text{GroundTruth}_t$$

**元素**:
- **蓝色茎 (Stem)**: 每个时间步的残差值
- **黑色水平线**: 零线 (参考)

**目的**: 
- 观察误差是否有**系统性偏差**（如相位偏移导致的余弦波形误差）
- 观察误差是否有**幅度偏差**（如预测值总是偏小或偏大）
- 判断误差的**一致性**（是否在某些部分更大）

### 5.3 误差曲线生成逻辑: 均值与标准差

**文件**: `mse_vs_epoch_{range}_range.png`

该图表用于展示模型**收敛情况**及**稳定性**。

#### 数据收集

对于每一个频率 $f$ 和每一个 epoch $e$：

1. 获取该频率、该范围的所有样本
2. 逐个预测，计算每个样本的 MSE：
   $$\text{MSE}_i = \text{Mean}[(\text{Pred}_i - \text{Truth}_i)^2]$$
3. 得到一个 MSE 列表：
   $$\text{MSE\_List} = [\text{MSE}_1, \text{MSE}_2, ..., \text{MSE}_N]$$

#### 统计计算

```python
mean = np.mean(MSE_List)       # 平均 MSE
std = np.std(MSE_List)         # 标准差
upper = mean + std             # 上界
lower = max(mean - std, 1e-6)  # 下界 (Log坐标防止负数)
```

#### 绘图规范

```
对于每个频率 f:
  X轴: Epoch (1, 5, 10, 20, 30, 40, 50)
  Y轴: MSE Loss (Log Scale, 对数坐标)
  
  如果 f in FREQS_TRAIN (10Hz, 20Hz):
    线条: ━━━━ 实线
    宽度: 2.5
    透明度: 0.9
    标记: ○
  
  否则 (其他频率):
    线条: - - - - 虚线
    宽度: 2.0
    透明度: 0.7
    标记: ○
  
  误差区间:
    颜色: 与曲线相同
    透明度: 0.2
    范围: [mean - std, mean + std]
```

**示意图**:

```
Log MSE
     │
10^0 ├─●─────●    ━━━ 10Hz (Train)
     │  │ ╱╲ │
     │  │╱  ╲│
10^-1├──●────●    - - 15Hz (Test)
     │   │╱╲│
     │   ●  ●    ━━━ 20Hz (Train)
10^-2│    │╱╲│
     │    ●  ●   - - 25Hz (Test)
     │
     └────────────── Epoch
       1  5 10 20 30 40 50
```

---

## 6. 运行指南

### 6.1 基本运行

```bash
cd d:\yangkeyin\code\softplus_activation_experiment

python PRPs\wave_seq2seq_freq_generalization_v3.py
```

### 6.2 输出示例

```
====================================================================================================
实验协议: Transformer 频率泛化能力测试 (Frequency Generalization Test) v3
====================================================================================================

[1/4] 生成全量波形数据...
  ✓ 生成了 6 个频率的波形，每个 5000 点

====================================================================================================
[2/4] 种子 1/3: 训练模型 (Seed=42)
====================================================================================================
  训练集大小: 6602 样本
  开始训练 50 个 epoch...
    Epoch   1/50 | Train Loss: 0.125634
    → 在 Epoch 1 进行详细评估...
    Epoch  10/50 | Train Loss: 0.023456
    → 在 Epoch 10 进行详细评估...
    ...
    Epoch  50/50 | Train Loss: 0.001234
  ✓ 种子 42 的训练完成

[3/4] 生成 MSE vs Epoch 曲线...
✓ 已保存: mse_vs_epoch_train_range.png
✓ 已保存: mse_vs_epoch_test_range.png

[4/4] 实验完成！
====================================================================================================
所有结果已保存到: ./figures/Freq_Generalization_Exp_20231128_143022
====================================================================================================
```

### 6.3 参数调整

如需修改实验配置，编辑脚本顶部的配置部分：

```python
# 频率设置
FREQS_TRAIN = [10.0, 20.0]
FREQS_TEST = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

# 序列参数
SEQ_LEN = 100
PRED_LEN = 100
SAMPLING_RATE = 200

# 数据切分
RANGE_TRAIN_END = 3500
RANGE_TEST_START = 3500

# 训练参数
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.001

# 可视化控制
VIS_EPOCHS = [1, 5, 10, 20, 30, 40, 50]
VIS_SAMPLE_INDICES = [0, 10, 20]
```

---

## 7. 预期结果

### 7.1 理想情况: 良好的泛化性

```
Train Range MSE vs Epoch          Test Range MSE vs Epoch
     │                                 │
10^0 ├────●────●    ━━ 10Hz          10^0 ├────●────●    ━━ 10Hz
     │    │╱ ╲ │                          │    │╱ ╲ │
10^-1├──●   ●─●    - - 15Hz         10^-1├──●   ●─●    - - 15Hz
     │   │╱ ╲│ │                          │   │╱ ╲│ │
     │   ●   ●  ━━ 20Hz                  │   ●   ●  ━━ 20Hz
10^-2├    │╱ ╲│                     10^-2├    │╱ ╲│
     │    ●   ●  - - 25Hz                │    ●   ●  - - 25Hz
     │                                    │
     └──────────── Epoch               └──────────── Epoch
```

**特征**:
- ✅ 训练频率 (10Hz, 20Hz) MSE 最小
- ✅ 插值频率 (15Hz, 25Hz) MSE 中等，增长平缓
- ✅ 外推频率 (5Hz, 30Hz) MSE 较大但不极端
- ✅ Test Range 损失略高于 Train Range（正常现象）
- ✅ 曲线趋势平稳，说明模型已收敛

**拟合图特征**:
- 预测信号与真实信号**高度重合**（红线贴近蓝线）
- 误差**均匀分布**在零线附近（无明显系统偏差）
- 误差**幅度小**（远小于信号幅度）

### 7.2 中等情况: 部分泛化

```
Train Range                       Test Range
10^0 ├────●                       10^0 ├────●
     │    │╲                           │    │╲╲
10^-1├──●  │ ╲ ━━ 10Hz           10^-1├──●  │  ╲ ━━ 10Hz
     │   │  │  ╲                      │   │  │   ╲
     │   ●  │   ╲━━ 20Hz              │   ●  │    ╲━━ 20Hz
10^-2├    │ │    ╲                10^-2├    │ │     ╲
     │    ● ●     ●- - 15Hz (插值)     │    ● ●      ●- - 15Hz
     │         ╲  ╱                    │        ╲   ╱
     │          ●────○ - - 25Hz (外推) │         ● ─○ - - 25Hz
     └────────────── Epoch              └────────────── Epoch
```

**特征**:
- ✅ 训练频率 (10Hz, 20Hz) 性能好
- ⚠️  插值频率 (15Hz, 25Hz) 损失明显上升
- ❌ 外推频率 (5Hz, 30Hz) 损失激增，甚至发散
- ⚠️  Test Range 损失与 Train Range 差异大，说明过拟合

**拟合图特征**:
- 插值频率的预测出现**相位偏移**或**幅度衰减**
- 外推频率的预测完全**失效**或出现**各种伪影**

### 7.3 差的情况: 完全无泛化

```
Train Range                       Test Range
10^0 ├────●                       10^0 ├────△
     │    │╲╲╲╲╲                      │    │╲╲╲╲╲
10^-1├──●  │   ╲━━ 10Hz          10^-1├──△  │    ╲━━ 10Hz
     │   │  │    ╲                    │   │  │     ╲
     │   ●  │     ●━━ 20Hz           │   △  │      △━━ 20Hz
10^-2├    │ │      ╲                10^-2├    │       ╲
     │    ● ●       △ - - 15Hz       │    △  △        △ - - 15Hz
     │         ╲  ╱                   │         ╲    ╱
     │          △─────△ - - 25Hz      │          △──△ - - 25Hz
     └────────────── Epoch            └────────────── Epoch
```

**特征**:
- ✅ 训练频率 (10Hz, 20Hz) 性能尚可
- ❌ 插值频率 (15Hz, 25Hz) 损失已经非常大
- ❌ 外推频率 (5Hz, 30Hz) 损失巨大，完全无法预测
- ❌ Test Range 损失远高于 Train Range，严重过拟合

**拟合图特征**:
- 所有非训练频率的预测都是**随机噪声**或**常数**
- 无法识别出波形的基本特征

---

## 8. 常见问题 (FAQ)

### Q1: 为什么要硬切分数据而不是随机分割？

**A**: 这是时间序列预测的标准做法。硬切分保证了：
- 训练和测试数据来自完全不同的时间段
- 避免了时间泄露（Temporal Leakage）
- 更真实地反映模型在未来数据上的性能

### Q2: 为什么在 Train Range 和 Test Range 都评估？

**A**: 这两个评估的目的不同：
- **Train Range**: 评估模型在训练数据时间段的性能（应该最好）
- **Test Range**: 评估模型在未来数据的泛化能力（更重要）

如果两者差异大，说明过拟合；如果两者都差，说明欠拟合。

### Q3: 为什么使用对数坐标绘制 MSE？

**A**: 
- MSE 的数值范围可能很大（从 0.0001 到 100）
- 对数坐标让曲线更易读
- 可以同时看到大数值和小数值的变化趋势

### Q4: 如何判断泛化性好不好？

**A**: 查看 `mse_vs_epoch_test_range.png` 图：
- **好**: 所有频率的 MSE 都在降低，曲线平缓
- **中等**: 训练频率 MSE 低，插值频率 MSE 中等，外推频率 MSE 高
- **差**: 非训练频率的 MSE 远高于训练频率，甚至发散

---

## 9. 完整检查清单

运行实验前的检查：

- [ ] Python 和 PyTorch 已安装
- [ ] 脚本路径正确 (`PRPs/wave_seq2seq_freq_generalization_v3.py`)
- [ ] 输出目录可写 (`./figures/`)
- [ ] GPU 可用或接受使用 CPU
- [ ] VIS_EPOCHS 和 VIS_SAMPLE_INDICES 设置合理

---

**版本**: v3.0  
**最后更新**: 2024-11-28  
**作者**: AI Assistant
