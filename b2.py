import numpy as np
import tkinter as tk
from tkinter import messagebox, LabelFrame

class LinearEquationSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Equation Solver")

        # Khung cho nhập số phương trình
        input_frame = LabelFrame(root, text="Nhập số phương trình/ẩn số", padx=10, pady=10)
        input_frame.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

        self.n_label = tk.Label(input_frame, text="Số phương trình/ẩn số (n):")
        self.n_label.grid(row=0, column=0)

        self.n_entry = tk.Entry(input_frame)
        self.n_entry.grid(row=0, column=1)

        # Khung cho ma trận và vector
        self.matrix_frame = LabelFrame(root, text="Ma trận hệ số và vector hằng số", padx=10, pady=10)
        self.matrix_frame.grid(row=1, column=0, padx=10, pady=10, columnspan=2)

        # Nút tạo ma trận
        button_frame = LabelFrame(root, text="Tác vụ", padx=10, pady=10)
        button_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        self.generate_button = tk.Button(button_frame, text="Tạo ma trận", command=self.generate_matrix_fields)
        self.generate_button.grid(row=0, column=0)

        self.reset_button = tk.Button(button_frame, text="Reset", command=self.reset_fields)
        self.reset_button.grid(row=0, column=1)

        self.solve_button = None
        self.solution_label = None
        self.matrix_entries = []
        self.constant_entries = []

    def generate_matrix_fields(self):
        # Xóa các ô nhập liệu cũ (nếu có)
        self.reset_fields(clear_n_entry=False)

        try:
            self.n = int(self.n_entry.get())
            if self.n <= 0:
                raise ValueError("Số n phải là số nguyên dương.")

            # Tạo các ô nhập cho ma trận hệ số và hằng số
            for i in range(self.n):
                row_entries = []
                for j in range(self.n):
                    entry = tk.Entry(self.matrix_frame, width=5)
                    entry.grid(row=i, column=j)
                    row_entries.append(entry)
                self.matrix_entries.append(row_entries)

                # Tạo ô nhập vector hằng số
                constant_entry = tk.Entry(self.matrix_frame, width=5)
                constant_entry.grid(row=i, column=self.n)
                self.constant_entries.append(constant_entry)

            # Nút giải hệ phương trình
            if not self.solve_button:
                self.solve_button = tk.Button(self.root, text="Giải", command=self.solve_system)
                self.solve_button.grid(row=3, column=0, columnspan=2, pady=10)

        except ValueError:
            messagebox.showerror("Lỗi nhập liệu", "Hãy nhập một số nguyên dương cho n.")

    def solve_system(self):
        try:
            A = np.array([[float(self.matrix_entries[i][j].get()) for j in range(self.n)] for i in range(self.n)])
            B = np.array([float(self.constant_entries[i].get()) for i in range(self.n)])

            rank_A = np.linalg.matrix_rank(A)
            augmented_matrix = np.column_stack((A, B))
            rank_augmented = np.linalg.matrix_rank(augmented_matrix)

            if rank_A < rank_augmented:
                solution_text = "Hệ phương trình vô nghiệm."
            elif rank_A == rank_augmented < self.n:
                solution_text = "Hệ phương trình có vô số nghiệm."
            else:
                X = np.linalg.solve(A, B)
                solution_text = "Nghiệm của hệ phương trình:\n" + "\n".join([f"x{i + 1} = {X[i]:.2f}" for i in range(self.n)])

            if self.solution_label:
                self.solution_label.destroy()
            self.solution_label = tk.Label(self.root, text=solution_text)
            self.solution_label.grid(row=4, column=0, columnspan=2, pady=10)

        except ValueError:
            messagebox.showerror("Lỗi", "Hãy nhập các giá trị hợp lệ.")
        except Exception:
            messagebox.showerror("Lỗi", "Có lỗi xảy ra. Hãy kiểm tra lại dữ liệu nhập.")

    def reset_fields(self, clear_n_entry=True):
        # Xóa các ô nhập liệu cũ
        for row in self.matrix_entries:
            for entry in row:
                entry.destroy()
        self.matrix_entries.clear()

        for entry in self.constant_entries:
            entry.destroy()
        self.constant_entries.clear()

        if self.solution_label:
            self.solution_label.destroy()

        if clear_n_entry:
            self.n_entry.delete(0, tk.END)

# Chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = LinearEquationSolverApp(root)
    root.mainloop()
