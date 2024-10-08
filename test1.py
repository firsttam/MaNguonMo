import numpy as np
import matplotlib.pyplot as plotter

# Thiết lập các thông số
samplingFrequency = 100
samplingInterval = 1 / samplingFrequency
beginTime = 0
endTime = 10
signal1Frequency = 4
signal2Frequency = 7

# Tạo các điểm thời gian
time = np.arange(beginTime, endTime, samplingInterval)

# Tạo hai sóng sine
amplitude1 = np.sin(2 * np.pi * signal1Frequency * time)
amplitude2 = np.sin(2 * np.pi * signal2Frequency * time)

# Tạo subplot
figure, axis = plotter.subplots(5, 1)
plotter.subplots_adjust(hspace=1)

# Biểu diễn miền thời gian cho sóng sine 1
axis[0].set_title('Sine wave with a frequency of 4 Hz')
axis[0].plot(time, amplitude1)
axis[0].set_xlabel('Time')
axis[0].set_ylabel('Amplitude')

# Biểu diễn miền thời gian cho sóng sine 2
axis[1].set_title('Sine wave with a frequency of 7 Hz')
axis[1].plot(time, amplitude2)
axis[1].set_xlabel('Time')
axis[1].set_ylabel('Amplitude')

# Cộng các sóng sine
amplitude = amplitude1 + amplitude2

# Biểu diễn miền thời gian của sóng sine tổng hợp
axis[2].set_title('Sine wave with multiple frequencies')
axis[2].plot(time, amplitude)
axis[2].set_xlabel('Time')
axis[2].set_ylabel('Amplitude')

# Biểu diễn miền tần số
fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Chuẩn hóa biên độ
tpCount = len(amplitude)
frequencies = np.fft.fftfreq(tpCount, d=samplingInterval)[:tpCount // 2]
fourierTransform = fourierTransform[:tpCount // 2]  # Chỉ lấy nửa đầu

# Lọc tần số 4 Hz
target_frequency = 4
frequency_resolution = frequencies[1] - frequencies[0]  # Độ phân giải tần số

# Tìm chỉ số tương ứng với tần số 4 Hz
target_index = np.argmin(np.abs(frequencies - target_frequency))
filtered_transform = np.zeros_like(fourierTransform)
filtered_transform[target_index] = fourierTransform[target_index]

# Chuyển đổi ngược về miền thời gian
filtered_amplitude = np.fft.ifft(filtered_transform * len(amplitude))

# Biểu diễn miền tần số
axis[3].set_title('Fourier transform depicting the frequency components')
axis[3].plot(frequencies, abs(fourierTransform))
axis[3].set_xlabel('Frequency')
axis[3].set_ylabel('Amplitude')

# Biểu diễn sóng sine đã lọc
axis[4].set_title('Filtered Signal (Only 4 Hz)')
axis[4].plot(time, filtered_amplitude.real)
axis[4].set_xlabel('Time')
axis[4].set_ylabel('Amplitude')

plotter.show()
