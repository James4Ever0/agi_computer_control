import tkinter as tk
from tkterminal import Terminal

root = tk.Tk()
terminal = Terminal(pady=5, padx=5)
terminal.shell=True
terminal.pack(expand=True, fill='both')
root.mainloop()
