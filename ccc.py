import tkinter as tk
from tkinter import filedialog


def choose_file(directory):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir=directory)
    return file_path


test_dir = r'sliwki_test'
file_path = choose_file(test_dir)
print(file_path)
