import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import mysql.connector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from level1 import get_data_1, load_data_1, save_data_1, plot_data_1, compute_linear_regression_1, get_determinant_1
from level2 import get_data_2, load_data_2, save_data_2, plot_data_2, compute_linear_regression_2, get_determinant_2
from level3 import get_data_3, load_data_3, save_data_3, plot_data_3, compute_linear_regression_3, get_determinant_3

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="SHanteaLeYERaERONEVEr",
        database="linearreg"
    )

class LinearRegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Regression App")
        self.root.geometry("900x700")

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TButton", font=("Arial", 12), padding=5)
        self.style.configure("TLabel", font=("Arial", 12))
        self.style.configure("TCombobox", font=("Arial", 12))

        self.root.configure(bg="#f0f8ff")

        self.level = tk.StringVar(value="level1")
        self.data = None

        self.create_header()

        self.create_level_selector()

        self.create_buttons()

        self.output = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=15, width=100, font=("Courier New", 10))
        self.output.pack(pady=10)

    def create_header(self):
        header_frame = tk.Frame(self.root, bg="#4682B4", pady=10)
        header_frame.pack(fill="x")
        tk.Label(header_frame, text="Linear Regression App", bg="#4682B4", fg="white", font=("Arial", 20, "bold")).pack()

    def create_level_selector(self):
        frame = tk.Frame(self.root, bg="#f0f8ff")
        frame.pack(pady=10, fill="x")

        ttk.Label(frame, text="Select Level:", background="#f0f8ff").pack(side="left", padx=5)
        self.level_dropdown = ttk.Combobox(frame, textvariable=self.level, values=["level1", "level2", "level3"])
        self.level_dropdown.pack(side="left", padx=5)
        self.level_dropdown.bind("<<ComboboxSelected>>", self.level_changed)

        ttk.Label(frame, text="Currently Selected:", background="#f0f8ff").pack(side="left", padx=10)
        self.selected_label = ttk.Label(frame, text=self.level.get(), font=("Arial", 12, "bold"), background="#f0f8ff")
        self.selected_label.pack(side="left")

    def level_changed(self, event=None):
        self.selected_label.config(text=self.level.get())
        self.log_output(f"Switched to {self.level.get()}.")

    def create_buttons(self):
        button_frame = tk.Frame(self.root, bg="#f0f8ff")
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Fetch Data", command=self.fetch_data).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Save Data", command=self.save_data).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Plot Data", command=self.plot_data).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Compute Regression", command=self.compute_regression).grid(row=0, column=3, padx=5)
        ttk.Button(button_frame, text="Load Data", command=self.load_data).grid(row=0, column=4, padx=5)

    def load_data(self):
        try:
            mydb = connect_db()
            cursor = mydb.cursor()
            if self.level.get() == "level1":
                self.data = load_data_1(mydb, cursor, "linearreg")
            elif self.level.get() == "level2":
                self.data = load_data_2(mydb, cursor, "linearreg")
            elif self.level.get() == "level3":
                self.data = load_data_3(mydb, cursor, "linearreg")
            else:
                raise ValueError("Invalid level selected.")
            self.log_output(f"Data loaded successfully for {self.level.get()}.")
            self.log_output(self.data.head().to_string())
            messagebox.showinfo("Success", f"Data loaded successfully for {self.level.get()}.")
        except Exception as e:
            self.log_output(f"Error loading data: {e}")
            messagebox.showerror("Error", f"Error loading data: {e}")

    def log_output(self, message):
        self.output.insert(tk.END, message + "\n")
        self.output.see(tk.END)

    def fetch_data(self):
        try:
            if self.level.get() == "level1":
                self.data = get_data_1()
            elif self.level.get() == "level2":
                self.data = get_data_2()
            elif self.level.get() == "level3":
                self.data = get_data_3([
                    "NY.GDP.PCAP.CD", 
                    "SL.TLF.CACT.MA.ZS",
                    "SL.TLF.CACT.FE.ZS", 
                    "SL.TLF.TOTL.FE.ZS",
                    "SE.ADT.LITR.MA.ZS",
                    "SE.ADT.LITR.FE.ZS",
                    "SP.DYN.LE00.MA.IN",
                    "SP.DYN.LE00.FE.IN"
                ])
            else:
                raise ValueError("Invalid level selected.")
            self.log_output(f"Data fetched successfully for {self.level.get()}.")
            self.log_output(self.data.head().to_string())
        except Exception as e:
            self.log_output(f"Error fetching data: {e}")
            messagebox.showerror("Error", f"Error fetching data: {e}")

    def save_data(self):
        try:
            mydb = connect_db()
            cursor = mydb.cursor()
            if self.level.get() == "level1":
                save_data_1(self.data, mydb, cursor, "linearreg")
            elif self.level.get() == "level2":
                save_data_2(self.data, mydb, cursor, "linearreg")
            elif self.level.get() == "level3":
                save_data_3(self.data, mydb, cursor, "linearreg")
            else:
                raise ValueError("Invalid level selected.")
            self.log_output(f"Data saved successfully to the database for {self.level.get()}.")
            messagebox.showinfo("Success", f"Data saved successfully for {self.level.get()}.")
        except Exception as e:
            self.log_output(f"Error saving data: {e}")
            messagebox.showerror("Error", f"Error saving data: {e}")

    def plot_data(self):
        try:
            if self.level.get() == "level1":
                plot_data_1(self.data)
            elif self.level.get() == "level2":
                plot_data_2(self.data)
            elif self.level.get() == "level3":
                plot_data_3(self.data)
            else:
                raise ValueError("Invalid level selected.")
        except Exception as e:
            self.log_output(f"Error plotting data: {e}")
            messagebox.showerror("Error", f"Error plotting data: {e}")

    def compute_regression(self):
        try:
            if self.level.get() == "level1":
                intercept, slope = compute_linear_regression_1(self.data)
                r_squared = get_determinant_1(self.data, slope, intercept)
            elif self.level.get() == "level2":
                coefficients = compute_linear_regression_2(self.data)
                r_squared = get_determinant_2(self.data, coefficients)
            elif self.level.get() == "level3":
                coefficients = compute_linear_regression_3(self.data)
                r_squared = get_determinant_3(self.data, coefficients)
            else:
                raise ValueError("Invalid level selected.")
            self.log_output(f"Regression Results for {self.level.get()}:\nR^2: {r_squared}")
            messagebox.showinfo("Regression Results", f"R^2: {r_squared}")
        except Exception as e:
            self.log_output(f"Error computing regression: {e}")
            messagebox.showerror("Error", f"Error computing regression: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LinearRegressionApp(root)
    root.mainloop()