import tkinter as tk
import numpy as np

def calculate():
    try:
        a = np.array(eval(entry_a.get()))
        b = np.array(eval(entry_b.get()))

        # Cộng
        result_add.set(str(a + b))

        # Trừ
        result_sub.set(str(a - b))

        # Nhân
        result_mul.set(str(np.dot(a, b)))

        # Chia (Nhân với nghịch đảo)
        if np.linalg.det(b) != 0:  # Kiểm tra xem ma trận b có nghịch đảo không
            result_div.set(str(np.dot(a, np.linalg.inv(b))))
        else:
            result_div.set("Ma trận B không có nghịch đảo")

        # Nghịch đảo
        if np.linalg.det(a) != 0:  # Kiểm tra xem ma trận a có nghịch đảo không
            result_inv.set(str(np.linalg.inv(a)))
        else:
            result_inv.set("Ma trận A không có nghịch đảo")

        # Hạng
        rank.set(str(np.linalg.matrix_rank(a)))

    except Exception as e:
        result_add.set(str(e))
        result_sub.set(str(e))
        result_mul.set(str(e))
        result_div.set(str(e))
        result_inv.set(str(e))
        rank.set(str(e))

def reset():
    entry_a.delete(0, tk.END)
    entry_b.delete(0, tk.END)
    result_add.set("")
    result_sub.set("")
    result_mul.set("")
    result_div.set("")
    result_inv.set("")
    rank.set("")

# Tạo cửa sổ GUI
root = tk.Tk()
root.title("Ma trận và Phép toán")

# Nhập ma trận A
tk.Label(root, text="Nhập ma trận A (vd: [[1, 2], [3, 4]]):").pack()
entry_a = tk.Entry(root, width=50)
entry_a.pack()

# Nhập ma trận B
tk.Label(root, text="Nhập ma trận B (vd: [[5, 6], [7, 8]]):").pack()
entry_b = tk.Entry(root, width=50)
entry_b.pack()

# Các biến để hiển thị kết quả
result_add = tk.StringVar()
result_sub = tk.StringVar()
result_mul = tk.StringVar()
result_div = tk.StringVar()
result_inv = tk.StringVar()
rank = tk.StringVar()

# Nút tính toán
tk.Button(root, text="Tính toán", command=calculate).pack()

# Nút reset
tk.Button(root, text="Reset", command=reset).pack()

# Hiển thị kết quả
tk.Label(root, text="Kết quả A + B:").pack()
tk.Label(root, textvariable=result_add).pack()

tk.Label(root, text="Kết quả A - B:").pack()
tk.Label(root, textvariable=result_sub).pack()

tk.Label(root, text="Kết quả A * B:").pack()
tk.Label(root, textvariable=result_mul).pack()

tk.Label(root, text="Kết quả A / B (A * B⁻¹):").pack()
tk.Label(root, textvariable=result_div).pack()

tk.Label(root, text="Nghịch đảo của A:").pack()
tk.Label(root, textvariable=result_inv).pack()

tk.Label(root, text="Hạng của A:").pack()
tk.Label(root, textvariable=rank).pack()

# Bắt đầu vòng lặp chính của GUI
root.mainloop()
