<<<<<<< HEAD
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Machine Learning Application")

# Biến toàn cục
df = None
model = None
mse = mae = rmse = None
y_test = y_predict = None


# Hàm chọn file
def load_file():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        messagebox.showinfo("Thông báo", "Dữ liệu đã được load thành công!")
    else:
        messagebox.showerror("Lỗi", "Không có file nào được chọn!")


# Hàm train mô hình
def train_model():
    global model, df, mse, mae, rmse, y_test, y_predict
    if df is None:
        messagebox.showerror("Lỗi", "Hãy chọn file dữ liệu trước!")
        return

    # Chọn thuật toán
    algorithm = algo_var.get()
    x = np.array(df.iloc[:, :-1]).astype(np.float64)
    y = np.array(df.iloc[:, -1]).astype(np.float64)

    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    if algorithm == "KNN":
        model = KNeighborsRegressor(n_neighbors=3)
    elif algorithm == "Hồi quy tuyến tính":
        model = LinearRegression()
    elif algorithm == "Cây quyết định":
        model = DecisionTreeRegressor()
    elif algorithm == "Vector hỗ trợ":
        model = SVR()

    # Train model
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    # Tính các chỉ số sai số
    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    rmse = np.sqrt(mse)

    # Hiển thị kết quả
    mse_label.config(text=f"MSE: {mse:.2f}")
    mae_label.config(text=f"MAE: {mae:.2f}")
    rmse_label.config(text=f"RMSE: {rmse:.2f}")

    # Vẽ đồ thị kết quả dự đoán
    plt.figure()
    plt.plot(range(0, len(y_test)), y_test, 'ro', label='Original data')
    plt.plot(range(0, len(y_predict)), y_predict, 'bo', label='Fitted line')
    for i in range(0, len(y_test)):
        plt.plot([i, i], [y_test[i], y_predict[i]], 'g')
    plt.title('Kết quả dự đoán')
    plt.legend()
    plt.show()


# Hàm dự đoán dữ liệu mới
def predict_new():
    if model is None:
        messagebox.showerror("Lỗi", "Hãy train mô hình trước!")
        return

    # Lấy dữ liệu mới
    new_data = new_data_entry.get()
    try:
        new_data = np.array([float(i) for i in new_data.split(',')]).reshape(1, -1)
        prediction = model.predict(new_data)
        result_label.config(text=f"Kết quả dự đoán: {prediction[0]:.2f}")
    except ValueError:
        messagebox.showerror("Lỗi", "Dữ liệu nhập không hợp lệ!")


# Hàm vẽ biểu đồ tỷ lệ sai số
def plot_error_distribution():
    if y_test is None or y_predict is None:
        messagebox.showerror("Lỗi", "Hãy train mô hình trước!")
        return

    # Tính sai số
    errors = abs(y_predict - y_test)

    # Phân loại sai số
    error_gt_2 = np.sum(errors > 2)
    error_between_1_2 = np.sum((errors >= 1) & (errors <= 2))
    error_lt_1 = np.sum(errors < 1)

    # Vẽ biểu đồ hình tròn
    labels = ['Sai số > 2 điểm', 'Sai số trong khoảng 1-2 điểm', 'Sai số < 1 điểm']
    sizes = [error_gt_2, error_between_1_2, error_lt_1]
    colors = ['red', 'yellow', 'green']
    plt.figure()
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Tỷ lệ sai số')
    plt.show()


# Hàm vẽ biểu đồ cột MSE, MAE, RMSE
def plot_error_bars():
    if mse is None or mae is None or rmse is None:
        messagebox.showerror("Lỗi", "Hãy train mô hình trước!")
        return

    # Vẽ biểu đồ cột
    labels = ['MSE', 'RMSE', 'MAE']
    values = [mse, rmse, mae]
    plt.figure()
    plt.bar(labels, values, color=['blue', 'orange', 'green'])
    plt.title('Biểu đồ cột cho MSE, RMSE, MAE')
    plt.ylabel('Giá trị lỗi')
    plt.show()


# Tạo layout giao diện
load_button = tk.Button(root, text="Chọn file", command=load_file)
load_button.pack()

algo_var = tk.StringVar()
algo_var.set("KNN")  # Giá trị mặc định
algorithms_menu = tk.OptionMenu(root, algo_var, "KNN", "Hồi quy tuyến tính", "Cây quyết định", "Vector hỗ trợ")
algorithms_menu.pack()

train_button = tk.Button(root, text="Train", command=train_model)
train_button.pack()

mse_label = tk.Label(root, text="MSE: N/A")
mse_label.pack()

mae_label = tk.Label(root, text="MAE: N/A")
mae_label.pack()

rmse_label = tk.Label(root, text="RMSE: N/A")
rmse_label.pack()

new_data_label = tk.Label(root, text="Nhập dữ liệu mới (phân tách bởi dấu phẩy):")
new_data_label.pack()

new_data_entry = tk.Entry(root)
new_data_entry.pack()

predict_button = tk.Button(root, text="Dự đoán", command=predict_new)
predict_button.pack()

result_label = tk.Label(root, text="Kết quả dự đoán: N/A")
result_label.pack()

# Nút vẽ biểu đồ tỷ lệ sai số
error_dist_button = tk.Button(root, text="Vẽ biểu đồ tỷ lệ sai số", command=plot_error_distribution)
error_dist_button.pack()

# Nút vẽ biểu đồ cột cho MSE, RMSE, MAE
error_bars_button = tk.Button(root, text="Vẽ biểu đồ cột MSE, RMSE, MAE", command=plot_error_bars)
error_bars_button.pack()

# Chạy ứng dụng
root.mainloop()
=======
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

# Yêu cầu người dùng nhập tần số cần lọc
target_frequency = float(input("Nhập tần số cần lọc (Hz): "))

# Biểu diễn miền tần số
fourierTransform = np.fft.fft(amplitude) / len(amplitude)  # Chuẩn hóa biên độ
tpCount = len(amplitude)
frequencies = np.fft.fftfreq(tpCount, d=samplingInterval)[:tpCount // 2]
fourierTransform = fourierTransform[:tpCount // 2]  # Chỉ lấy nửa đầu

# Tạo bộ lọc tần số
filtered_transform = np.zeros_like(fourierTransform)

# Tìm chỉ số tương ứng với tần số cần lọc
target_index = np.argmin(np.abs(frequencies - target_frequency))

# Chỉ giữ lại thành phần tần số mong muốn
filtered_transform[target_index] = fourierTransform[target_index]

# Chuyển đổi ngược về miền thời gian
filtered_amplitude = np.fft.ifft(filtered_transform * len(amplitude))

# Biểu diễn miền tần số
axis[3].set_title('Fourier transform depicting the frequency components')
axis[3].plot(frequencies, abs(fourierTransform))
axis[3].set_xlabel('Frequency (Hz)')
axis[3].set_ylabel('Amplitude')

# Biểu diễn sóng sine đã lọc
axis[4].set_title(f'Filtered Signal (Only {target_frequency} Hz)')
shortened_time = time[:len(filtered_amplitude.real)]
axis[4].plot(shortened_time, filtered_amplitude.real)
axis[4].set_xlabel('Time')
axis[4].set_ylabel('Amplitude')

plotter.show()
>>>>>>> b68f1c72d95ae6a392fb0e85abf4bdb1c8bd4fb0
