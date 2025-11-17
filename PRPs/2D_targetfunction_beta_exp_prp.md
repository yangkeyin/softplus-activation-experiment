# **实验提案 (PRP): 2D目标函数下Beta参数影响的普适性 (V2)**

## **(基于 new\_scripts/run\_beta.py 框架)**

## **1\. 实验目标**

本实验的核心目标是严格遵循 new\_scripts/run\_beta.py 的实验框架，将其扩展到二维目标函数，以验证 beta 参数对高维函数拟合影响的普适性。

我们将通过以下步骤实现这一目标：

1. **维度扩展**：将目标函数从 y \= sin(x) 扩展为 z \= sin(x) \* sin(y)。  
2. **模型扩展**：将 FNNModel 修改为 FNNModel2D，使其接受2D输入 (x, y)。  
3. **数据修改**：遵循用户指导“训练点每1/2pi取一个”。  
   * 对于 \[-2π, 2π\] 的范围，每个维度有 $4\\pi / (0.5\\pi) \+ 1 \= 9$ 个点。  
   * 总训练点将构成一个 $9 \\times 9 \= 81$ 的网格。  
   * 测试点将使用一个更密集的网格（例如 $50 \\times 50$）以便于热力图可视化。  
4. **基线对比 (新增)**：**添加“三次样条插值 (Cubic Spline)”** 作为非神经网络的基线，并将其 test\_rms 绘制在最终的误差图上。  
5. **指标对齐**：复现 run\_beta.py 中的核心时域指标（train\_rms, test\_rms）。  
   * **关键修改**：由于 get\_fq\_coef 频谱分析工具仅适用于1D函数，本实验将**移除所有与 spectrum\_error 相关的计算和绘图**。  
6. **可视化修改**：  
   * 将 run\_beta.py 中的 plot\_each\_epoch (2D线图) 修改为 plot\_each\_epoch\_2D (使用热力图)。  
   * 将 run\_beta.py 中的 beta\_rms\_seed.png (3x1 布局) 修改为 (2x1 布局)，仅保留 train\_rms 和 test\_rms 的演化图，并**加入样条插值基线**。

## **2\. 核心逻辑 (规划 new\_scripts/run\_beta\_2D.py 脚本)**

此 PRP 详细规划了新脚本 new\_scripts/run\_beta\_2D.py 的实现细节。

### **A. 依赖 (Dependencies)**

* torch, numpy, matplotlib  
* from scipy.interpolate import RectBivariateSpline (**新增**)  
* src/utils.py (仅用于 set\_seed，不再需要 get\_fq\_coef)

### **B. 关键组件修改**

#### **1\. 新增： generate\_data\_2D 函数**

def generate\_data\_2D(data\_range, train\_points\_per\_dim, test\_points\_per\_dim, device):  
    """  
    生成2D网格数据  
    """  
    \# 1\. 训练数据 (9x9 网格)  
    x\_ticks\_train \= np.linspace(data\_range\[0\], data\_range\[1\], train\_points\_per\_dim)  
    y\_ticks\_train \= np.linspace(data\_range\[0\], data\_range\[1\], train\_points\_per\_dim)  
    \# ⭐️ 修正：使用 'ij' 索引以匹配 RectBivariateSpline 的 z\[i, j\] \= f(x\[i\], y\[j\]) 期望  
    xx\_train, yy\_train \= np.meshgrid(x\_ticks\_train, y\_ticks\_train, indexing='ij')   
    X\_train\_np \= np.vstack(\[xx\_train.ravel(), yy\_train.ravel()\]).T  
    y\_train\_np \= np.sin(X\_train\_np\[:, 0\]) \* np.sin(X\_train\_np\[:, 1\])  
      
    \# 2\. 测试数据 (50x50 网格)  
    x\_ticks\_test \= np.linspace(data\_range\[0\], data\_range\[1\], test\_points\_per\_dim)  
    y\_ticks\_test \= np.linspace(data\_range\[0\], data\_range\[1\], test\_points\_per\_dim)  
    \# ⭐️ 修正：同样使用 'ij' 索引  
    xx\_test, yy\_test \= np.meshgrid(x\_ticks\_test, y\_ticks\_test, indexing='ij')  
    X\_test\_np \= np.vstack(\[xx\_test.ravel(), yy\_test.ravel()\]).T  
    y\_test\_np \= np.sin(X\_test\_np\[:, 0\]) \* np.sin(X\_test\_np\[:, 1\])

    \# 3\. 转换为张量  
    X\_train \= torch.tensor(X\_train\_np, dtype=torch.float32).to(device)  
    y\_train \= torch.tensor(y\_train\_np, dtype=torch.float32).reshape(-1, 1).to(device)  
    X\_test \= torch.tensor(X\_test\_np, dtype=torch.float32).to(device)  
    y\_test \= torch.tensor(y\_test\_np, dtype=torch.float32).reshape(-1, 1).to(device)  
      
    \# 返回 (X\_test, y\_test) 和 (xx\_test, yy\_test) 等以便绘图  
    \# ⭐️ 新增：返回 1D 坐标轴和 2D 训练网格  
    return X\_train, y\_train, X\_test, y\_test, \\  
           x\_ticks\_train, y\_ticks\_train, y\_train\_np.reshape(train\_points\_per\_dim, train\_points\_per\_dim), \\  
           xx\_test, yy\_test

#### **2\. 新增： FNNModel2D 类**

class FNNModel2D(nn.Module):  
    def \_\_init\_\_(self, n, beta):  
        super().\_\_init\_\_()  
        self.layers \= nn.Sequential(  
            nn.Linear(2, n), \# 接受 (x, y) 两个输入  
            nn.Softplus(beta=beta),  
            nn.Linear(n, 1\)  
        )  
    def forward(self, x):  
        return self.layers(x)

#### **3\. 修改： main 函数核心**

* **导入**：在脚本顶部添加 from scipy.interpolate import RectBivariateSpline。  
* **数据**：  
  * 调用 generate\_data\_2D 并接收所有返回值：  
    X\_train, y\_train, X\_test, y\_test, x\_ticks\_train, y\_ticks\_train, y\_train\_grid\_np, xx\_test, yy\_test \= generate\_data\_2D(DATA\_RANGE, 9, 50, DEVICE)  
* **新增：计算样条插值**：在 generate\_data\_2D 之后立即添加：  
  \# \-----------------------------------------------------  
  \# ⭐️ 新增：计算三次样条插值 (Cubic Spline)  
  \# \-----------------------------------------------------  
  print("Calculating Cubic Spline Interpolation baseline...")  
  \# 1\. 创建 RectBivariateSpline 插值器  
  \# (由于 generate\_data\_2D 已使用 'ij' 索引, y\_train\_grid\_np 无需转置)  
  spline\_interpolator \= RectBivariateSpline(x\_ticks\_train, y\_ticks\_train, y\_train\_grid\_np, kx=3, ky=3)

  \# 2\. 在测试集(50x50)的坐标上进行插值  
  X\_test\_np \= X\_test.cpu().numpy()  
  y\_test\_np\_flat \= y\_test.cpu().numpy().flatten()

  \# 3\. 逐点评估 (grid=False)  
  y\_pred\_spline\_flat \= spline\_interpolator(X\_test\_np\[:, 0\], X\_test\_np\[:, 1\], grid=False)

  \# 4\. 计算样条插值的 Test RMS  
  spline\_test\_rms \= np.sqrt(np.mean((y\_pred\_spline\_flat \- y\_test\_np\_flat)\*\*2))  
  print(f"Cubic Spline Test RMS: {spline\_test\_rms:.6f}")  
  \# \-----------------------------------------------------

* **移除频谱**：删除对 get\_fq\_coef 和 true\_coef 的所有调用。  
* **模型**：实例化 model \= FNNModel2D(n=100, beta=beta)。  
* **结果字典**：在 results\_base 字典中添加样条插值结果：  
  results\_base \= {  
      \# ... (x\_train, y\_train, x\_test, y\_test) ...  
      'xx\_train\_grid': xx\_train, \# 保存网格用于绘图  
      'yy\_train\_grid': yy\_train,  
      'xx\_test\_grid': xx\_test,  
      'yy\_test\_grid': yy\_test,  
      'spline\_test\_rms': spline\_test\_rms, \# ⭐️ 新增  
      'metrics': {beta: {} for beta in BETA}  
  }

* **指标计算**：在训练循环的评估步骤中，**只计算** train\_rms 和 test\_rms。移除 spectrum\_error。  
* **数据保存**：results 字典中将不再包含 pred\_coef 和 spectrum\_error。同时需要保存 y\_pred (训练集预测) 和 y\_pred\_test (测试集预测)。  
* **绘图**：  
  * 调用 plot\_each\_epoch\_2D(...) (见 4.A)。  
  * 修改 beta\_rms\_seed.png 的绘图逻辑 (见 4.B)。

## **3\. 关键指标 (Key Metrics)**

与 run\_beta.py 保持一致，但移除频谱：

1. **train\_rms**: 训练集上的均方根误差。  
2. **test\_rms**: 测试集上的均方根误差。

## **4\. 可视化方案 (Visualization Plan)**

可视化输出将严格模仿 run\_beta.py 的结构，但进行2D适配。

### **A. 图 1: 逐 Epoch 分析图 (热力图)**

* **函数**: plot\_each\_epoch\_2D(...)  
* **布局**: 3x3 网格 (扩展布局以展示训练集和测试集)  
* **内容**:  
  * **行 0: 训练集 (**$9 \\times 9$**)**  
    * \[0, 0\]: **真实训练 Z**。使用 imshow 显示 y\_train 在 $9 \\times 9$ 网格上的热力图。  
    * \[0, 1\]: **平均预测 Z (训练集)**。使用 imshow 显示跨 seed 平均的 y\_pred\_train 热力图。  
    * \[0, 2\]: **平均训练误差**。使用 imshow 显示 (y\_pred\_train \- y\_train) 的平均热力图。  
  * **行 1: 测试集 (**$50 \\times 50$**)**  
    * \[1, 0\]: **真实测试 Z**。使用 imshow 显示 y\_test 在 $50 \\times 50$ 网格上的热力图。  
    * \[1, 1\]: **平均预测 Z (测试集)**。使用 imshow 显示跨 seed 平均的 y\_pred\_test 热力图。  
    * \[1, 2\]: **平均测试误差**。使用 imshow 显示 (y\_pred\_test \- y\_test) 的平均热力图。  
  * **行 2: (留空或用于其他诊断)**  
    * \[2, 0\]: (留空)  
    * \[2, 1\]: (留空)  
    * \[2, 2\]: (留空)  
* **标题**: 每个子图将包含 AVG RMS: ... (如果适用)。  
* **目的**: 宏观上**可视化** beta 值对训练集拟合和测试集泛化的影响。

### **B. 图 2: 误差演化总结图 (Error Evolution Summary)**

* **文件名**: beta\_rms\_seed.png  
* **布局**: 2x1 网格 (替换 run\_beta.py 的 3x1 网格)。  
* **内容**:  
  * **子图 \[0\]**: **Train RMS vs Epoch**。与 run\_beta.py 完全一致的线图。  
  * **子图 \[1\]**: **Test RMS vs Epoch**。与 run\_beta.py 完全一致的线图。  
  * **(移除)**: Spectrum Error RMS vs Epoch 子图被移除。  
  * **⭐️ 新增**: 在子图 \[1\] (Test RMS) 中，添加一条**红色水平虚线**，表示 spline\_test\_rms 的值，并更新图例。  
* **目的**: **量化** beta 对学习速度和最终收敛精度的影响，并与三次样条插值基线进行对比。

## **5\. 预期产物 (Deliverables)**

1. **Python 脚本**: new\_scripts/run\_beta\_2D.py (根据本 PRP 规划创建)。  
2. **实验结果**:  
   * figures/beta\_2D\_experiment/results\_base\_2D.pkl (包含 train\_rms, test\_rms 和 spline\_test\_rms 的结果字典)。  
3. **可视化图表**:  
   * figures/beta\_2D\_experiment/beta\_1.0/beta\_1.0\_epoch\_1000.png (3x3 热力图)。  
   * figures/beta\_2D\_experiment/beta\_4.0/beta\_4.0\_epoch\_1000.png (3x3 热力图)。  
   * ... (其他 beta 和 epoch)  
   * figures/beta\_2D\_experiment/beta\_rms\_seed.png (2x1 演化图，**包含样条插值基线**)。