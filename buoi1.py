import tkinter as tk
from tkinter import messagebox
import numpy as np

# Khởi tạo mảng dữ liệu với mã sinh viên và điểm
student_ids = np.array([], dtype=int)
student_scores = np.zeros((0, 3))  # 3 môn học

# Hàm thêm sinh viên mới
def add_student():
    global student_ids, student_scores
    student_id = int(entry_student_id.get())
    
    if student_id in student_ids:
        messagebox.showerror("Error", "Mã sinh viên đã tồn tại.")
    else:
        
        student_ids = np.append(student_ids, student_id)
        student_scores = np.vstack([student_scores, np.zeros(3)])
        messagebox.showinfo("Thành công", f"Đã thêm sinh viên {student_id}")
        entry_student_id.delete(0, tk.END)

# Hàm nhập điểm
def input_scores():
    student_id = int(entry_student_id.get())
    score1 = float(entry_score1.get())
    score2 = float(entry_score2.get())
    score3 = float(entry_score3.get())

    if student_id not in student_ids:
        messagebox.showerror("Error", "Mã sinh viên không tồn tại.")
    else:
        idx = np.where(student_ids == student_id)[0][0]
        student_scores[idx] = [score1, score2, score3]
        messagebox.showinfo("Thành công", f"Đã nhập điểm cho sinh viên {student_id}")
        entry_score1.delete(0, tk.END)
        entry_score2.delete(0, tk.END)
        entry_score3.delete(0, tk.END)

# Hàm hiển thị điểm sinh viên
def display_scores():
    display_window = tk.Toplevel(root)
    display_window.title("Danh sách sinh viên")

    text = tk.Text(display_window, width=40, height=15)
    text.pack()

    text.insert(tk.END, "Mã SV | Môn 1 | Môn 2 | Môn 3\n")
    text.insert(tk.END, "---------------------------\n")
    for i, student_id in enumerate(student_ids):
        scores = student_scores[i]
        text.insert(tk.END, f"{student_id:<6} | {scores[0]:<5} | {scores[1]:<5} | {scores[2]:<5}\n")

# Hàm tính điểm trung bình của sinh viên
def average_score():
    student_id = int(entry_student_id.get())
    if student_id not in student_ids:
        messagebox.showerror("Error", "Mã sinh viên không tồn tại.")
    else:
        idx = np.where(student_ids == student_id)[0][0]
        avg = np.mean(student_scores[idx])
        messagebox.showinfo("Điểm trung bình", f"Điểm trung bình của SV {student_id}: {avg:.2f}")

# Tạo giao diện với Tkinter
root = tk.Tk()
root.title("Quản lý điểm sinh viên")

# Khung nhập mã sinh viên
label_student_id = tk.Label(root, text="Mã sinh viên")
label_student_id.grid(row=0, column=0)
entry_student_id = tk.Entry(root)
entry_student_id.grid(row=0, column=1)

# Khung nhập điểm
label_score1 = tk.Label(root, text="Điểm môn 1")
label_score1.grid(row=1, column=0)
entry_score1 = tk.Entry(root)
entry_score1.grid(row=1, column=1)

label_score2 = tk.Label(root, text="Điểm môn 2")
label_score2.grid(row=2, column=0)
entry_score2 = tk.Entry(root)
entry_score2.grid(row=2, column=1)

label_score3 = tk.Label(root, text="Điểm môn 3")
label_score3.grid(row=3, column=0)
entry_score3 = tk.Entry(root)
entry_score3.grid(row=3, column=1)

# Các nút chức năng
btn_add_student = tk.Button(root, text="Thêm sinh viên", command=add_student)
btn_add_student.grid(row=4, column=0, pady=5)

btn_input_scores = tk.Button(root, text="Nhập điểm", command=input_scores)
btn_input_scores.grid(row=4, column=1, pady=5)

btn_display_scores = tk.Button(root, text="Hiển thị danh sách", command=display_scores)
btn_display_scores.grid(row=5, column=0, pady=5)

btn_avg_score = tk.Button(root, text="Tính điểm trung bình", command=average_score)
btn_avg_score.grid(row=5, column=1, pady=5)

# Chạy giao diện
root.mainloop()
