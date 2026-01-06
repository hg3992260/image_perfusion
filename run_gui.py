import os
import sys

# Ensure src is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

from tkinterdnd2 import TkinterDnD
from src.gui.app import PerfusionApp

def main():
    try:
        root = TkinterDnD.Tk()
        # Set icon if available
        # if os.path.exists("icon.ico"): root.iconbitmap("icon.ico")
        
        app = PerfusionApp(root)
        root.mainloop()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
