#!/bin/bash

# 脚本名称: ex01_limit_beta_test.sh
# 目的: 对比不同初始化 std 下，稀疏数据的拟合平滑度
# 日期: 2025-01-05

# 定义常量
EPOCHS=10000
POINTS=500   # 点数
WIDTH=1024
BETA=100
DEGREE=300
SAVE_DIR="../experiments_results/limit_test_beta"

echo "=== 开始 Std 对比实验 ==="

# 1. 小 Std (0.01) - 预期: 学习缓慢，非常平滑，甚至欠拟合
echo "Running Small Std..."
python ../new_scripts/Limit_Test_Beta_Refractor.py \
    --train_points $POINTS --width $WIDTH --epochs $EPOCHS --beta $BETA \
    --degree $DEGREE \
    --init_std 0.01 \
    --save_dir $SAVE_DIR \
    --exp_name "std_small"

# 2. 中 Std (1.0) - 预期: 标准表现
echo "Running default Std..."
python ../new_scripts/Limit_Test_Beta_Refractor.py \
    --train_points $POINTS --width $WIDTH --epochs $EPOCHS --beta $BETA \
    --degree $DEGREE \
    --save_dir $SAVE_DIR \
    --exp_name "std_medium"

# 3. 大 Std (10.0) - 预期: 初始权重混乱，可能导致过拟合或高频震荡
echo "Running Large Std..."
python ../new_scripts/Limit_Test_Beta_Refractor.py \
    --train_points $POINTS --width $WIDTH --epochs $EPOCHS --beta $BETA \
    --degree $DEGREE \
    --init_std 10.0 \
    --save_dir $SAVE_DIR \
    --exp_name "std_large"

echo "=== 实验结束. 结果保存在 $SAVE_DIR ==="