amplitude_Fredformer.py的频谱分析一致性修正。主要修改包括：

1. 频谱分析核心修改 （train_and_analyze函数）：
   
   - 添加了Hanning窗函数应用于所有时域信号（x_test_sample、y_test_sample、pred_y_time），有效抑制了频谱泄露
   - 对短信号（y_test_sample、pred_y_time）进行零填充到统一长度（192），确保频率分辨率一致
   - 计算统一长度的FFT，保证频谱比较的公平性
   - 使用统一的频率轴参数（freq_axis_common）替换了原来的freq_axis_input和freq_axis_pred
2. 绘图函数更新 ：
   
   - 修改了run_figure_2a函数中的左右两个振幅图，使用统一的频率轴参数
   - 修改了run_figure_2b函数中的三个子图，同样使用统一的频率轴参数

- 解决频率不匹配问题 ：

- 在 run_figure_2a 函数中添加了频率换算逻辑，将分析目标频率(k)转换为适合N=10000数据生成的"工厂频率"： data_gen_freqs_left = [(k / SEQ_LEN) * 10000 for k in key_freqs_2a] 和 data_gen_freqs_right = [(k / SEQ_LEN) * 10000 for k in key_freqs_2a]
- 在 run_figure_2b 函数中添加了类似的频率换算逻辑： data_gen_freqs = [(k / SEQ_LEN) * 10000 for k in key_freqs_2b]
- 修改了所有 generate_time_series 调用，使用换算后的频率值
- 保留了 train_and_analyze 调用中使用原始 key_freqs_2a 和 key_freqs_2b 作为分析目标
- 解决不公平的加窗比较问题 ：

- 实现了"先填充，后加窗"的正确分析流程
- 为所有信号定义了统一的N=192窗函数： win_common = np.hanning(SEQ_LEN)
- 先对短信号进行零填充： true_padded = np.pad(y_test_sample, (0, SEQ_LEN - PRED_LEN), 'constant') 和 pred_padded = np.pad(pred_y_time, (0, SEQ_LEN - PRED_LEN), 'constant')
- 然后对所有填充后的信号应用同一个窗函数
- 计算统一长度的FFT并使用统一的频率轴

v3
目标： 修改 train_and_analyze 函数的绘图分析部分，使其在 N=96 的尺度上进行公平比较。

定位文件： amplitude_Fredformer.py，函数 train_and_analyze。

找到此函数的末尾，"# --- 4. 准备最终绘图数据 ---" 之后的代码块。

删除该区域的所有FFT和绘图数据逻辑（从 print("应用 '先填充，后加窗'...") 到 plot_data = {...}）。

替换为以下最终正确的逻辑：

Python

# <--- 最终修正：在 N=PRED_LEN 的尺度上进行“公平”比较 ---
print("应用 N=PRED_LEN 尺度进行公平绘图分析...")

# 1. 定义 N=PRED_LEN 的窗函数
win_common = np.hanning(PRED_LEN) # 窗长度为 96

# 2. 提取 N=PRED_LEN 的信号片段
# Input = 拿 lookback 窗口的最后 96 个点
input_segment = x_test_sample[-PRED_LEN:] 
# Ground Truth = 拿 预测窗口的 96 个点
true_segment = y_test_sample
# Forecasting = 拿 预测结果的 96 个点
pred_segment = pred_y_time

# 3. 对所有 N=96 的片段应用同一个窗
input_windowed = input_segment * win_common
true_windowed = true_segment * win_common
pred_windowed = pred_segment * win_common

# 4. 计算 FFT 幅度 (所有都是 N=PRED_LEN)
input_fft_mag = np.abs(np.fft.fft(input_windowed))
true_fft_mag = np.abs(np.fft.fft(true_windowed))
pred_fft_mag = np.abs(np.fft.fft(pred_windowed))

# 5. 定义统一的频率轴 (基于 PRED_LEN)
fft_freq_common = np.fft.fftfreq(PRED_LEN, d=1)

# 6. 取FFT的前半部分 (N/2)
N_common = PRED_LEN // 2

# 7. 修正 key_frequencies，使其适应 N=PRED_LEN
# 我们的 k=10, 25, 40 是相对于 N=SEQ_LEN (192) 定义的
# 换算到 N=PRED_LEN (96) 的 k'
# k' = f * N' = (k / N) * N' = k * (N' / N) = k * (96 / 192) = k / 2
key_freqs_pred_k = [k * (PRED_LEN / SEQ_LEN) for k in key_frequencies]


plot_data = {
    'input_fft_mag': input_fft_mag[:N_common],
    'true_fft_mag': true_fft_mag[:N_common], 
    'pred_fft_mag': pred_fft_mag[:N_common], 
    'freq_axis_common': fft_freq_common[:N_common] * PRED_LEN, # 转换为 k (0 到 48)
    'key_freqs_plot_k': key_freqs_pred_k 
}
# <--- 修正结束 ---
修改绘图函数 (run_figure_2a 和 run_figure_2b)：

删除 common_freq_axis_left = plot_left['freq_axis_common']

替换所有 ax.plot(common_freq_axis_left, ...) 为 ax.plot(plot_left['freq_axis_common'], ...)

替换所有 ax.set_xlim(2, max(key_freqs_k) * 1.5) 为 ax.set_xlim(2, max(plot_left['key_freqs_plot_k']) * 1.5)

修改X轴标签：ax.set_xlabel("F (Frequency Component k) on common N=96 axis")