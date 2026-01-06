#!/bin/bash

# 脚本名称: ex01_run_layers.sh
# 目的: 对比不同mlp layers 数量下，频谱特性的变化
# 日期: 2025-01-06

COMMON_ARGS="--train_points 500 --width 128  --beta 4.0 --epochs 10000 --save_dir ./experiments_results/layer_analysis"

echo "--- Running Layer=2 (Baseline) ---"
python ./new_scripts/run_layers.py $COMMON_ARGS --num_layers 2 --exp_name "L2_base"

echo "--- Running Layer=3 ---"
python ./new_scripts/run_layers.py $COMMON_ARGS --num_layers 3 --exp_name "L3_deep"

echo "--- Running Layer=5 ---"
python ./new_scripts/run_layers.py $COMMON_ARGS --num_layers 5 --exp_name "L5_deep"

echo "--- Running Layer=8 (Very Deep) ---"
python ./new_scripts/run_layers.py $COMMON_ARGS --num_layers 8 --exp_name "L8_deep"

# layers 的数量应该放在title上
# fit title要去掉Linear Interpolation，加上layer数量，去掉beta
# title得重新设计
# parameter数量 也要记录下来，放到title上
# Experiment Finished. Saved to ../experiments_results/layer_analysis\20260106_115324_L5_deep_Layers5_Beta4.0_N500_Width1024
# --- Running Layer=8 (Very Deep) ---

# Model Structure Check:
# MLP(
#   (net): Sequential(
#     (0): Linear(in_features=1, out_features=1024, bias=True)
#     (1): Softplus(beta=4.0, threshold=20.0)
#     (2): Linear(in_features=1024, out_features=1024, bias=True)
#     (3): Softplus(beta=4.0, threshold=20.0)
#     (4): Linear(in_features=1024, out_features=1024, bias=True)
#     (5): Softplus(beta=4.0, threshold=20.0)
#     (6): Linear(in_features=1024, out_features=1024, bias=True)
#     (7): Softplus(beta=4.0, threshold=20.0)
#     (8): Linear(in_features=1024, out_features=1024, bias=True)
#     (9): Softplus(beta=4.0, threshold=20.0)
#     (10): Linear(in_features=1024, out_features=1024, bias=True)
#     (11): Softplus(beta=4.0, threshold=20.0)
#     (12): Linear(in_features=1024, out_features=1024, bias=True)
#     (13): Softplus(beta=4.0, threshold=20.0)
#     (14): Linear(in_features=1024, out_features=1, bias=True)
#   )
# )
# Total Parameter Count: 6300673
# ------------------------------
# Start Training: 20260106_115723_L8_deep_Layers8_Beta4.0_N500_Width1024
# Epoch 0, Loss: 9.994440
# Epoch 2000, Loss: 9.980000
# Epoch 4000, Loss: 9.980000
# Epoch 6000, Loss: 9.980000
# Epoch 8000, Loss: 9.980000
# Raw results saved to ../experiments_results/layer_analysis\20260106_115723_L8_deep_Layers8_Beta4.0_N500_Width1024\results.npy
# Final Result: Linear MSE=0.019344, MLP MSE=9.998000
# Experiment Finished. Saved to ../experiments_results/layer_analysis\20260106_115723_L8_deep_Layers8_Beta4.0_N500_Width1024
# (neurosort)
# 图片变空白，loss没有下降是什么原因
