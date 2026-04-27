import tkinter as tk
from tkinter import filedialog, messagebox
import os
from annotation import process_images


class CropperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Annotation Tool")
        self.root.geometry("500x300")

        self.input_dir = ""

        # === Title ===
        tk.Label(root, text="Select Image Folder", font=("Arial", 12)).pack(pady=10)

        # === Input path field ===
        self.input_entry = tk.Entry(root, width=50)
        self.input_entry.pack(pady=5)

        # === Browse button ===
        tk.Button(root, text="Browse", command=self.select_folder).pack()

        # Spacer
        tk.Label(root, text="").pack()

        # === Bounding box expansion input ===
        tk.Label(root, text="Bounding Box Expansion (%) (default: 10, recommended: 10–15)").pack()

        self.extend_entry = tk.Entry(root, width=10)
        self.extend_entry.pack()
        self.extend_entry.insert(0, "10")

        # Spacer
        tk.Label(root, text="").pack()

        # === Start button ===
        tk.Button(
            root,
            text="Start Annotation",
            bg="green",
            fg="white",
            command=self.start_processing
        ).pack(pady=10)

    def select_folder(self):
        """Open folder dialog and update input field"""
        self.input_dir = filedialog.askdirectory(title="Select Image Folder")
        if self.input_dir:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, self.input_dir)

    def start_processing(self):
        """Validate input and start annotation pipeline"""
        extend_percentage = self.extend_entry.get()

        # Validate expansion percentage
        try:
            extend_percentage = float(extend_percentage)
            if not (0 <= extend_percentage <= 100):
                raise ValueError
        except:
            messagebox.showerror("Error", "Please enter a number between 0 and 100")
            return

        self.input_dir = self.input_entry.get()

        # Validate input directory
        if not self.input_dir:
            messagebox.showerror("Error", "Please select an image folder")
            return

        # Run processing (output = same directory)
        process_images(self.input_dir, self.input_dir, extend_percentage)

        messagebox.showinfo("Done", "Annotation completed! XML files generated.")


if __name__ == "__main__":
    root = tk.Tk()
    app = CropperApp(root)
    root.mainloop()