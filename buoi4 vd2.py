import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import numpy as np


# Hàm tính diện tích và chu vi của các hình
def calculate():
    shape = shape_var.get()
    mode = mode_var.get()

    if mode == "2D":
        if shape == "Hình tròn":
            radius = float(param_entry.get())
            area = np.pi * radius ** 2
            perimeter = 2 * np.pi * radius
            draw_circle(radius)

        elif shape == "Hình tam giác":
            a, b, c = map(float, param_entry.get().split(","))
            # Kiểm tra điều kiện tồn tại của tam giác
            if a + b > c and a + c > b and b + c > a:
                s = (a + b + c) / 2
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                perimeter = a + b + c
                draw_triangle(a, b, c)
            else:
                messagebox.showerror("Lỗi", "Không tồn tại tam giác.")
                return

        elif shape == "Hình tứ giác":
            try:
                a, b, c, d = map(float, param_entry.get().split(","))
                messagebox.showinfo("Thông báo", "Hình phức tạp không thể tính khi chỉ có tham số cạnh.")
                return
            except ValueError:
                messagebox.showerror("Lỗi", "Vui lòng nhập 4 cạnh cho hình tứ giác.")
                return

        elif shape == "Hình thang":
            a, b, h = map(float, param_entry.get().split(","))
            area = (a + b) * h / 2
            perimeter = "Cần thêm thông tin để tính chu vi"
            draw_trapezoid(a, b, h)

        elif shape == "Hình chữ nhật":
            a, b = map(float, param_entry.get().split(","))
            area = a * b
            perimeter = 2 * (a + b)
            draw_rectangle(a, b)

        elif shape == "Hình vuông":
            a = float(param_entry.get())
            area = a ** 2
            perimeter = 4 * a
            draw_square(a)

        elif shape == "Hình bình hành":
            a, b, h = map(float, param_entry.get().split(","))
            area = a * h
            perimeter = 2 * (a + b)
            draw_parallelogram(a, b, h)

    else:
        messagebox.showinfo("Thông báo", "Chế độ 3D chưa được hỗ trợ trong phiên bản này.")
        return

    messagebox.showinfo("Kết quả", f"Diện tích: {area}\nChu vi: {perimeter}")


# Hàm vẽ hình
def draw_circle(radius):
    circle = plt.Circle((0, 0), radius, color='blue', fill=False)
    fig, ax = plt.subplots()
    ax.add_artist(circle)
    ax.set_xlim(-radius - 1, radius + 1)
    ax.set_ylim(-radius - 1, radius + 1)
    ax.set_aspect('equal', adjustable='box')
    plt.title("Hình tròn")
    plt.grid()
    plt.show()


def draw_triangle(a, b, c):
    # Giả sử a, b, c là ba cạnh của tam giác
    A = np.array([0, 0])
    B = np.array([a, 0])
    # Sử dụng định lý Cosine để tìm đỉnh C
    angle_C = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
    C = np.array([b * np.cos(angle_C), b * np.sin(angle_C)])

    triangle = np.array([A, B, C, A])

    plt.plot(triangle[:, 0], triangle[:, 1], color='blue')
    plt.fill(triangle[:, 0], triangle[:, 1], alpha=0.3)
    plt.xlim(-1, a + 1)
    plt.ylim(-1, b + 1)
    plt.title("Hình tam giác")
    plt.grid()
    plt.axis('equal')
    plt.show()


def draw_rectangle(a, b):
    rectangle = plt.Rectangle((0, 0), a, b, fill=None, edgecolor='blue')
    fig, ax = plt.subplots()
    ax.add_patch(rectangle)
    ax.set_xlim(-1, a + 1)
    ax.set_ylim(-1, b + 1)
    ax.set_aspect('equal', adjustable='box')
    plt.title("Hình chữ nhật")
    plt.grid()
    plt.show()


def draw_square(a):
    square = plt.Rectangle((0, 0), a, a, fill=None, edgecolor='blue')
    fig, ax = plt.subplots()
    ax.add_patch(square)
    ax.set_xlim(-1, a + 1)
    ax.set_ylim(-1, a + 1)
    ax.set_aspect('equal', adjustable='box')
    plt.title("Hình vuông")
    plt.grid()
    plt.show()


def draw_parallelogram(a, b, h):
    # Vẽ hình bình hành
    plt.plot([0, a, a + b, b, 0], [0, 0, h, h, 0], color='blue')
    plt.fill([0, a, a + b, b], [0, 0, h, h], alpha=0.3)
    plt.xlim(-1, a + b + 1)
    plt.ylim(-1, h + 1)
    plt.title("Hình bình hành")
    plt.grid()
    plt.show()


def draw_trapezoid(a, b, h):
    # Vẽ hình thang
    plt.plot([0, a, b, 0], [0, 0, h, h], color='blue')
    plt.fill([0, a, b, 0], [0, 0, h, h], alpha=0.3)
    plt.xlim(-1, max(a, b) + 1)
    plt.ylim(-1, h + 1)
    plt.title("Hình thang")
    plt.grid()
    plt.show()


# Tạo giao diện GUI
root = tk.Tk()
root.title("Ứng dụng Hình học")

# Chọn chế độ
mode_var = tk.StringVar(value="2D")
ttk.Label(root, text="Chọn chế độ:").grid(column=0, row=0, padx=10, pady=10)
ttk.Radiobutton(root, text="2D", variable=mode_var, value="2D").grid(column=1, row=0, padx=10, pady=10)
ttk.Radiobutton(root, text="3D", variable=mode_var, value="3D").grid(column=2, row=0, padx=10, pady=10)

# Chọn loại hình
shape_var = tk.StringVar(value="Hình tròn")
ttk.Label(root, text="Chọn hình:").grid(column=0, row=1, padx=10, pady=10)
shapes = ["Hình tròn", "Hình tam giác", "Hình tứ giác", "Hình thang", "Hình chữ nhật", "Hình vuông", "Hình bình hành"]
ttk.Combobox(root, textvariable=shape_var, values=shapes, state="readonly").grid(column=1, row=1, padx=10, pady=10)

# Nhập tham số
ttk.Label(root, text="Nhập tham số (cách nhau bằng dấu phẩy):").grid(column=0, row=2, padx=10, pady=10)
param_entry = ttk.Entry(root, width=40)
param_entry.grid(column=1, row=2, padx=10, pady=10)

# Nút tính toán
calculate_button = ttk.Button(root, text="Tính toán", command=calculate)
calculate_button.grid(column=0, row=3, columnspan=3, pady=10)

# Bắt đầu giao diện
root.mainloop()
