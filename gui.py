import tkinter as tk
from tkinter import messagebox, ttk
import time
import csv
import os  
from config import AUTHORIZED_PLATES



class GarageDoorGUI:
    def __init__(self, master, arduino, garage_entries):
        self.master = master
        self.master.title("Garage Door System")
        self.master.geometry("600x400")

        self.garage_entries = garage_entries

        # for the record list Treeview
        self.columns = ("Plate", "Time", "Status")
        self.tree = ttk.Treeview(self.master, columns=self.columns, show="headings")
        self.tree.heading("Plate", text="Plate")
        self.tree.heading("Time", text="Time")
        self.tree.heading("Status", text="Status")
        self.tree.pack(fill=tk.BOTH, expand=True, pady=10)

        # interface for the manual entry
        manual_frame = tk.Frame(self.master)
        manual_frame.pack(pady=10)

        manual_label = tk.Label(manual_frame, text="Manual Plate:")
        manual_label.pack(side=tk.LEFT, padx=5)

        self.manual_entry = tk.Entry(manual_frame)
        self.manual_entry.pack(side=tk.LEFT, padx=5)

        manual_entry_button = tk.Button(manual_frame, text="Enter", command=self.manual_entry_open)
        manual_entry_button.pack(side=tk.LEFT, padx=5)

        manual_exit_button = tk.Button(manual_frame, text="Exit", command=self.manual_exit)
        manual_exit_button.pack(side=tk.LEFT, padx=5)

        # Save as CSV button
        manual_save_button = tk.Button(manual_frame, text="Save as CSV", command=self.manual_save)
        manual_save_button.pack(side=tk.LEFT, padx=5)

    def add_entry(self, plate, status):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.tree.insert("", tk.END, values=(plate, current_time, status))
    def handle_entry(self, plate, source="Camera"):
        plate = plate.strip().upper()

        if plate in self.garage_entries:
          self.add_entry(plate, "Already Inside")
          messagebox.showwarning("Warning", f"{plate} is already inside!")
          return

        self.garage_entries.add(plate)
        self.add_entry(plate, "Authorized Entry")

    def manual_entry_open(self):
       manual_plate = self.manual_entry.get().strip().upper()

       if not manual_plate:
        messagebox.showwarning("Warning", "Please enter a valid plate!")
        return
       
       #unauthorized plate
       if manual_plate not in AUTHORIZED_PLATES:
        self.add_entry(manual_plate, "Unauthorized Access")
        messagebox.showerror("Unauthorized", f"{manual_plate} is NOT authorized!")
        return

       self.handle_entry(manual_plate, source="Manual")


    def manual_exit(self):
        manual_plate = self.manual_entry.get().strip().upper()
        if manual_plate:
            if manual_plate in self.garage_entries:
                self.garage_entries.remove(manual_plate)
                self.add_entry(manual_plate, "Exit")
               
                messagebox.showinfo("Success", f"Garage door opened for exit: {manual_plate}")
            else:
                messagebox.showwarning("Warning", f"Plate {manual_plate} is not currently inside!")
        else:
            messagebox.showwarning("Warning", "Please enter a valid plate!")

    def manual_save(self):
        try:
            # desktop  
            desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
            file_path = os.path.join(desktop_path, "garage_entries.csv")

            with open(file_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Plate", "Time", "Status"]) 
                for row_id in self.tree.get_children():
                    row = self.tree.item(row_id)['values']
                    writer.writerow(row)

            messagebox.showinfo("Success", f"Data saved successfully to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {e}")