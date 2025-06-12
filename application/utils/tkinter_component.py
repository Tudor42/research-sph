import socket
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from application.client.remote_state_manager import RemoteStateManager

def open_case_file():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="Open SPH case", filetypes=[("Python", "*.py"),("All","*.*")]
    )
    root.destroy()
    return path or None


def get_connection():
    root = tk.Tk()
    root.title("Connect")

    w, h = 300, 160
    ws, hs = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(ws-w)//2}+{(hs-h)//2}")

    host_var = tk.StringVar(value="127.0.0.1")
    port_var = tk.StringVar(value="50007")
    password_var = tk.StringVar()

    result = {"sock": None, "password": None}

    def on_connect():
        host = host_var.get().strip()
        try:
            port = int(port_var.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Port must be an integer")
            return

        pwd = password_var.get()

        try:
            sm = RemoteStateManager(host, port, pwd)
            if "error" in sm.state:
                messagebox.showerror("Server Error", sm.state["error"])
                return

            result["sm"] = sm
            messagebox.showinfo("Success", f"Connected to {host}:{port}")
            root.destroy()
        except Exception as e:
            messagebox.showerror("Connection Failed", str(e))

    tk.Label(root, text="Host:").grid(row=0, column=0, padx=8, pady=4, sticky="e")
    tk.Entry(root, textvariable=host_var).grid(row=0, column=1, padx=8, pady=4)

    tk.Label(root, text="Port:").grid(row=1, column=0, padx=8, pady=4, sticky="e")
    tk.Entry(root, textvariable=port_var).grid(row=1, column=1, padx=8, pady=4)

    tk.Label(root, text="Password:").grid(row=2, column=0, padx=8, pady=4, sticky="e")
    tk.Entry(root, textvariable=password_var, show="•").grid(row=2, column=1, padx=8, pady=4)

    tk.Button(root, text="Connect", command=on_connect).grid(
        row=3, column=0, columnspan=2, pady=10
    )

    root.mainloop()
    return result["sm"]
