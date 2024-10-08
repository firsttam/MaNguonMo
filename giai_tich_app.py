import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
import math

def calculate_2d(shape, dimensions):
    if shape == 'Triangle':
        a, b, c = dimensions
        s = (a + b + c) / 2
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        perimeter = a + b + c
    elif shape == 'Square':
        a = dimensions[0]
        area = a ** 2
        perimeter = 4 * a
    elif shape == 'Rectangle':
        a, b = dimensions
        area = a * b
        perimeter = 2 * (a + b)
    elif shape == 'Trapezoid':
        a, b, h = dimensions
        area = ((a + b) / 2) * h
        perimeter = None  # Requires additional inputs for full perimeter
    else:
        area = perimeter = None

    return area, perimeter

def get_dimensions(shape, dim_type):
    if dim_type == '2D':
        if shape == 'Triangle':
            a = simpledialog.askfloat("Input", "Enter side a:")
            b = simpledialog.askfloat("Input", "Enter side b:")
            c = simpledialog.askfloat("Input", "Enter side c:")
            return (a, b, c)
        elif shape == 'Square':
            a = simpledialog.askfloat("Input", "Enter side length:")
            return (a,)
        elif shape == 'Rectangle':
            a = simpledialog.askfloat("Input", "Enter length:")
            b = simpledialog.askfloat("Input", "Enter width:")
            return (a, b)
        elif shape == 'Trapezoid':
            a = simpledialog.askfloat("Input", "Enter base a:")
            b = simpledialog.askfloat("Input", "Enter base b:")
            h = simpledialog.askfloat("Input", "Enter height:")
            return (a, b, h)

def draw_shape_2d(shape, dimensions):
    if shape == 'Triangle':
        a, b, c = dimensions
        # Calculate the triangle vertices using a simple method
        vertices = np.array([[0, 0], [a, 0], [b/2, (c**2 - (b/2)**2)**0.5]])
        plt.fill(vertices[:, 0], vertices[:, 1], 'b', alpha=0.5)
        plt.xlim(-1, max(a, b) + 1)
        plt.ylim(-1, max(c, 5) + 1)
        plt.title('Triangle')
    elif shape == 'Square':
        a = dimensions[0]
        vertices = np.array([[0, 0], [a, 0], [a, a], [0, a]])
        plt.fill(vertices[:, 0], vertices[:, 1], 'g', alpha=0.5)
        plt.xlim(-1, a + 1)
        plt.ylim(-1, a + 1)
        plt.title('Square')
    elif shape == 'Rectangle':
        a, b = dimensions
        vertices = np.array([[0, 0], [a, 0], [a, b], [0, b]])
        plt.fill(vertices[:, 0], vertices[:, 1], 'r', alpha=0.5)
        plt.xlim(-1, a + 1)
        plt.ylim(-1, b + 1)
        plt.title('Rectangle')
    elif shape == 'Trapezoid':
        a, b, h = dimensions
        vertices = np.array([[0, 0], [a, 0], [a - (a - b)/2, h], [(b - (a - b)/2), h]])
        plt.fill(vertices[:, 0], vertices[:, 1], 'y', alpha=0.5)
        plt.xlim(-1, max(a, b) + 1)
        plt.ylim(-1, h + 1)
        plt.title('Trapezoid')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.show()

def main():
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính

    dim_type = simpledialog.askstring("Input", "Choose dimension (2D or 3D):", initialvalue="2D")
    if dim_type == '2D':
        shape = simpledialog.askstring("Input", "Choose shape (Triangle, Square, Rectangle, Trapezoid):", initialvalue="Triangle")
        dimensions = get_dimensions(shape, dim_type)
        area, perimeter = calculate_2d(shape, dimensions)
        draw_shape_2d(shape, dimensions)

        messagebox.showinfo("Result", f"Area: {area}\nPerimeter: {perimeter}")
    elif dim_type == '3D':
        # Chức năng 3D có thể được phát triển thêm ở đây
        messagebox.showinfo("Info", "3D shapes functionality is not implemented yet.")

if __name__ == "__main__":
    main()
