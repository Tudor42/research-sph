import tkinter as _tk
from tkinter import filedialog

def open_case_file():
    root = _tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="Open SPH case", filetypes=[("Python", "*.py"),("All","*.*")]
    )
    root.destroy()
    return path or None