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