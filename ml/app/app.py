import tkinter as tk

def on_button_click():
    label.config(text="Button Clicked!")

# Create the main window
root = tk.Tk()
root.title("Tkinter Example")

# Create a label
label = tk.Label(root, text="Hello, Tkinter!")
label.pack(pady=10)

# Create a button
button = tk.Button(root, text="Click me!", command=on_button_click)
button.pack(pady=10)

# Start the main event loop
root.mainloop()