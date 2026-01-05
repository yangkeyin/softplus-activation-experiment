# 实验提案：mlp网络深度（layers）与频谱特性的关系

**创建时间**： 2025-01-05
**状态**： 【Draft】
**分支**：feat/layer-spectrum (待创建)

## 核心假设
老师提出：增加网络深度会导致拟合函数的频谱学习变快（衰减斜率变小）
即：
**深层网络比浅层网络更倾向于学习高频写好**（在相同width和Epoch下）

## 实验变量
* **自变量**： numlayers = [2,3,5,8]
注：目前的mlp类不支持动态层数，需要重构模型定义

* **控制变量**：
    * 目标函数： frequencies[5, 15, 30, 50, 80]组成的sin（f*pi*x）混合信号
    * train_points = 500(np.linspace(-1,1,500))
    * test_points = 4000(np.random.uniform(-1,1,4000))
    * width = 1024
    * epochs = 10000
    * beta = 4.0

## 实现计划
需要创建一个新脚本 `scripts/run_layer_experiment.py`，关键修改点：
```python
layers = []
layers.append(Linear(1, width))
for _ in range(num_layers - 2):
    layers.append(Softplus(beta))
    layers.append(Linear(width, width))
layers.append(Softplus(beta))
layers.append(Linear(width, 1))
self.net = Sequential(*layers)
```

## 注意
1. 实验结果保存路径要自动生成，包含numlayers、width、beta等参数.可参考
```python
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{args.exp_name}_N{args.train_points}_Width{args.width}_Std{args.init_std}"
    output_path = os.path.join(args.save_dir, run_id)。
```
2. 每个实验结果要保存config.json
3. 参数通过parser输入
许多代码尽可能参考文件Limit_Test_Beta_Refractor.py