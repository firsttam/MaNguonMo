import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog


def load_data(file_path):
    """Load data from a CSV file into a numpy array."""
    try:
        data = np.genfromtxt(file_path, delimiter=',', dtype=str, encoding='utf-8', skip_header=1)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return np.array([])


def search_student(data, student_id):
    """Search for a student's information by ID."""
    if data.size == 0:
        return "Dữ liệu không được tải."

    student_data = data[data[:, 0] == student_id]
    if student_data.size == 0:
        return f"Không tìm thấy thông tin cho sinh viên có ID {student_id}."
    else:
        return "\n".join([", ".join(row) for row in student_data])


def search_subject(data, subject_name):
    """Search for grades of a specific subject."""
    if data.size == 0:
        return "Dữ liệu không được tải."

    subject_data = data[data[:, 2] == subject_name]
    if subject_data.size == 0:
        return f"Không tìm thấy điểm cho môn học {subject_name}."
    else:
        return "\n".join([f"ID: {row[0]}, Tên: {row[1]}, Điểm: {row[3]}" for row in subject_data])


def calculate_average(data, student_id):
    """Calculate the average grade for a specific student using numpy."""
    if data.size == 0:
        return "Dữ liệu không được tải."

    student_data = data[data[:, 0] == student_id]
    if student_data.size == 0:
        return f"Không tìm thấy thông tin cho sinh viên có ID {student_id}."
    else:
        try:
            grades = student_data[:, 3].astype(float)  # Convert grades to float
            average_grade = np.mean(grades)
            return f"Trung bình cộng điểm của sinh viên có ID {student_id} là {average_grade:.2f}."
        except ValueError:
            return "Có lỗi khi chuyển đổi điểm sang số thực. Vui lòng kiểm tra dữ liệu."


def search_action():
    choice = choice_var.get()
    student_id = id_entry.get()
    subject_name = subject_entry.get()

    if choice == '1':  # Tìm kiếm thông tin sinh viên
        result = search_student(data, student_id)
    elif choice == '2':  # Tìm kiếm điểm môn học
        result = search_subject(data, subject_name)
    elif choice == '3':  # Tính TBC điểm của sinh viên
        result = calculate_average(data, student_id)
    else:
        result = "Lựa chọn không hợp lệ."

    messagebox.showinfo("Kết quả", result)
def find_highest_and_lowest_average(data):
    """Find the student with the highest and lowest average grade."""
    if data.size == 0:
        return "Dữ liệu không được tải."

    try:
        student_ids = np.unique(data[:, 0])  # Lấy danh sách ID sinh viên duy nhất
        averages = []

        for student_id in student_ids:
            student_data = data[data[:, 0] == student_id]
            grades = student_data[:, 3].astype(float)
            average_grade = np.mean(grades)
            averages.append((student_id, student_data[0, 1], average_grade))  # Lưu trữ ID, tên và điểm trung bình

        highest = max(averages, key=lambda x: x[2])  # Tìm sinh viên có điểm trung bình cao nhất
        lowest = min(averages, key=lambda x: x[2])   # Tìm sinh viên có điểm trung bình thấp nhất

        result = (f"Sinh viên có điểm trung bình cao nhất: ID {highest[0]}, Tên {highest[1]}, "
                  f"Điểm trung bình {highest[2]:.2f}.\n"
                  f"Sinh viên có điểm trung bình thấp nhất: ID {lowest[0]}, Tên {lowest[1]}, "
                  f"Điểm trung bình {lowest[2]:.2f}.")
        return result

    except ValueError:
        return "Có lỗi khi xử lý dữ liệu. Vui lòng kiểm tra dữ liệu."


def search_action():
    choice = choice_var.get()
    student_id = id_entry.get()
    subject_name = subject_entry.get()

    if choice == '1':  # Tìm kiếm thông tin sinh viên
        result = search_student(data, student_id)
    elif choice == '2':  # Tìm kiếm điểm môn học
        result = search_subject(data, subject_name)
    elif choice == '3':  # Tính TBC điểm của sinh viên
        result = calculate_average(data, student_id)
    elif choice == '4':  # Tìm điểm trung bình cao nhất và thấp nhất
        result = find_highest_and_lowest_average(data)
    else:
        result = "Lựa chọn không hợp lệ."

    messagebox.showinfo("Kết quả", result)



def main():
    global data
    file_path = 'data.csv'  # Đặt đường dẫn đến file dữ liệu của bạn
    data = load_data(file_path)

    # Tạo cửa sổ chính
    root = tk.Tk()
    root.title("Tìm kiếm thông tin sinh viên")

    # Thêm các widget
    tk.Label(root, text="Chọn hành động:").pack(pady=5)

    global choice_var
    choice_var = tk.StringVar(value='1')

    tk.Radiobutton(root, text="Tìm kiếm thông tin sinh viên", variable=choice_var, value='1').pack(anchor='w')
    tk.Radiobutton(root, text="Tìm kiếm điểm môn học", variable=choice_var, value='2').pack(anchor='w')
    tk.Radiobutton(root, text="Tính TBC điểm của sinh viên", variable=choice_var, value='3').pack(anchor='w')
    tk.Radiobutton(root, text="Tìm điểm trung bình cao nhất và thấp nhất", variable=choice_var, value='4').pack(anchor='w')

    tk.Label(root, text="ID sinh viên:").pack(pady=5)
    global id_entry
    id_entry = tk.Entry(root)
    id_entry.pack(pady=5)

    tk.Label(root, text="Tên môn học (nếu có):").pack(pady=5)
    global subject_entry
    subject_entry = tk.Entry(root)
    subject_entry.pack(pady=5)

    tk.Button(root, text="Tìm kiếm", command=search_action).pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
