import numpy as np
import tkinter as tk
from tkinter import messagebox

class LinearEquationSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Giải hệ phương trình tuyến tính")

        # Hướng dẫn nhập n
        self.instructions_label = tk.Label(root, text="Nhập số phương trình/ẩn số (n):")
        self.instructions_label.grid(row=0, column=0, columnspan=2)

        # Ô nhập n
        self.n_entry = tk.Entry(root)
        self.n_entry.grid(row=0, column=2)

        # Nút để tạo các ô nhập liệu cho ma trận và vector hằng số
        self.generate_button = tk.Button(root, text="Tạo ma trận", command=self.generate_matrix_fields)
        self.generate_button.grid(row=0, column=3)

        # Nút reset để xóa các ô nhập liệu và nhập lại số phương trình
        self.reset_button = tk.Button(root, text="Reset", command=self.reset_fields)
        self.reset_button.grid(row=0, column=4)

        # Danh sách các ô nhập liệu cho ma trận và hằng số
        self.matrix_entries = []
        self.constant_entries = []
        self.solution_label = None
        self.solve_button = None
        self.label_a = None
        self.label_b = None
        self.separator_label = None

    def generate_matrix_fields(self):
        # Xóa các ô nhập liệu cũ (nếu có)
        self.reset_fields(clear_n_entry=False)

        try:
            # Lấy số lượng phương trình/ẩn số
            self.n = int(self.n_entry.get())
            if self.n <= 0:
                raise ValueError("Số n phải là số nguyên dương.")

            # Tạo nhãn cho Ma trận hệ số (A) và Hằng số (B) nếu chưa tồn tại
            if self.label_a:
                self.label_a.destroy()
            if self.label_b:
                self.label_b.destroy()
            if self.separator_label:
                self.separator_label.destroy()

            self.label_a = tk.Label(self.root, text="Ma trận hệ số (A)")
            self.label_a.grid(row=1, column=0, columnspan=self.n)

            # Separator " | " between matrix A and vector B
            self.separator_label = tk.Label(self.root, text=" | ")
            self.separator_label.grid(row=1, column=self.n)

            self.label_b = tk.Label(self.root, text="Hằng số (B)")
            self.label_b.grid(row=1, column=self.n + 1)

            # Tạo các ô nhập liệu cho ma trận hệ số và hằng số
            for i in range(self.n):
                row_entries = []
                for j in range(self.n):
                    entry = tk.Entry(self.root, width=5)
                    entry.grid(row=i + 2, column=j)
                    row_entries.append(entry)
                self.matrix_entries.append(row_entries)

                # Tạo ô nhập hằng số (B)
                constant_entry = tk.Entry(self.root, width=5)
                constant_entry.grid(row=i + 2, column=self.n + 1)
                self.constant_entries.append(constant_entry)

            # Nút để giải hệ phương trình
            self.solve_button = tk.Button(self.root, text="Giải", command=self.solve_system)
            self.solve_button.grid(row=self.n + 2, column=0, columnspan=self.n + 2)

        except ValueError:
            messagebox.showerror("Lỗi nhập liệu", "Hãy nhập một số nguyên dương cho n.")

    def solve_system(self):
        try:
            # Lấy dữ liệu từ các ô nhập và tạo ma trận A và vector B
            A = np.array([[float(self.matrix_entries[i][j].get()) for j in range(self.n)] for i in range(self.n)])
            B = np.array([float(self.constant_entries[i].get()) for i in range(self.n)])

            # Kiểm tra rank của ma trận A và (A|B) để xác định tính chất của hệ phương trình
            rank_A = np.linalg.matrix_rank(A)
            augmented_matrix = np.column_stack((A, B))
            rank_augmented = np.linalg.matrix_rank(augmented_matrix)

            if rank_A < rank_augmented:
                # Vô nghiệm (No solution)
                solution_text = "Hệ phương trình vô nghiệm."
            elif rank_A == rank_augmented < self.n:
                # Vô số nghiệm (Infinitely many solutions)
                solution_text = "Hệ phương trình có vô số nghiệm."
            else:
                # Giải hệ phương trình khi có nghiệm duy nhất
                X = np.linalg.solve(A, B)
                solution_text = "Nghiệm của hệ phương trình:\n" + "\n".join([f"x{i + 1} = {X[i]:.2f}" for i in range(self.n)])

            # Hiển thị kết quả
            if self.solution_label:
                self.solution_label.destroy()
            self.solution_label = tk.Label(self.root, text=solution_text)
            self.solution_label.grid(row=self.n + 3, column=0, columnspan=self.n + 2)

        except ValueError as e:
            messagebox.showerror("Lỗi", str(e))
        except Exception:
            messagebox.showerror("Lỗi", "Hãy nhập các giá trị số hợp lệ.")

    def reset_fields(self, clear_n_entry=True):
        # Clear all input fields and reset the interface
        for row in self.matrix_entries:
            for entry in row:
                entry.destroy()
        self.matrix_entries.clear()

        for entry in self.constant_entries:
            entry.destroy()
        self.constant_entries.clear()

        # Clear solution label and solve button if they exist
        if self.solution_label:
            self.solution_label.destroy()
            self.solution_label = None

        if self.solve_button:
            self.solve_button.destroy()
            self.solve_button = None

        # Clear the labels if they exist
        if self.label_a:
            self.label_a.destroy()
            self.label_a = None

        if self.label_b:
            self.label_b.destroy()
            self.label_b = None

        if self.separator_label:
            self.separator_label.destroy()
            self.separator_label = None

        # Clear the entry field for n if specified
        if clear_n_entry:
            self.n_entry.delete(0, tk.END)

# Khởi tạo giao diện
if __name__ == "__main__":
    root = tk.Tk()
    app = LinearEquationSolverApp(root)
    root.mainloop()
