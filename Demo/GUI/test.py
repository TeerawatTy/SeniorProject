import tkinter as tk
import customtkinter as ctk

# Main Application
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Minimal Test")
app.geometry("300x200")

# Create a basic option menu
options = ["Option 1", "Option 2", "Option 3"]

# Initialize the StringVar
selected_option = tk.StringVar(value=options[0])  # Set default option

# Create the CTkOptionMenu
option_menu = ctk.CTkOptionMenu(app, selected_option, *options)
option_menu.pack(pady=20)

app.mainloop()
