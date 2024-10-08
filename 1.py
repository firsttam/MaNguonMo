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
mse_results = {}
mae_results = {}
rmse_results = {}
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


# Hàm train mô hình với nhiều thuật toán
def train_model():
  global df, model, mse_results, mae_results, rmse_results, y_test, y_predict
  if df is None:
    messagebox.showerror("Lỗi", "Hãy chọn file dữ liệu trước!")
    return

  # Các thuật toán cần so sánh
  algorithms = {"KNN": KNeighborsRegressor(n_neighbors=3), "Hồi quy tuyến tính": LinearRegression(),
    "Cây quyết định": DecisionTreeRegressor(), "Vector hỗ trợ": SVR()}

  x = np.array(df.iloc[:, :-1]).astype(np.float64)
  y = np.array(df.iloc[:, -1]).astype(np.float64)

  # Chia dữ liệu train/test
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

  # Huấn luyện mô hình và tính toán các chỉ số sai số
  mse_results.clear()
  mae_results.clear()
  rmse_results.clear()

  for name, model in algorithms.items():
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)

    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    rmse = np.sqrt(mse)

    mse_results[name] = mse
    mae_results[name] = mae
    rmse_results[name] = rmse

  messagebox.showinfo("Thông báo", "Training hoàn tất!")
  plot_error_comparison()  # Vẽ biểu đồ so sánh sau khi train xong


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
  if not mse_results or not mae_results or not rmse_results:
    messagebox.showerror("Lỗi", "Hãy train mô hình trước!")
    return

  # Vẽ biểu đồ cột
  labels = ['MSE', 'RMSE', 'MAE']
  values = [mse_results["KNN"], rmse_results["KNN"], mae_results["KNN"]]
  plt.figure()
  plt.bar(labels, values, color=['blue', 'orange', 'green'])
  plt.title('Biểu đồ cột cho MSE, RMSE, MAE với KNN')
  plt.ylabel('Giá trị lỗi')
  plt.show()


# Hàm vẽ biểu đồ so sánh MSE, RMSE, MAE
def plot_error_comparison():
  if not mse_results or not mae_results or not rmse_results:
    messagebox.showerror("Lỗi", "Không có dữ liệu để vẽ biểu đồ!")
    return

  labels = list(mse_results.keys())
  mse_values = list(mse_results.values())
  mae_values = list(mae_results.values())
  rmse_values = list(rmse_results.values())

  # Vẽ biểu đồ cột so sánh MSE, RMSE, MAE
  x = np.arange(len(labels))  # Vị trí các nhãn trên trục x
  width = 0.2  # Độ rộng của mỗi cột

  fig, ax = plt.subplots()
  ax.bar(x - width, mse_values, width, label='MSE')
  ax.bar(x, rmse_values, width, label='RMSE')
  ax.bar(x + width, mae_values, width, label='MAE')

  ax.set_xlabel('Thuật toán')
  ax.set_title('So sánh MSE, RMSE, MAE giữa các thuật toán')
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.legend()

  plt.show()


# Tạo layout giao diện
load_button = tk.Button(root, text="Chọn file", command=load_file)
load_button.pack()

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
