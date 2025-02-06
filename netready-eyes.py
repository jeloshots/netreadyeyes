import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
from PIL import Image, ImageTk
import numpy as np
import wmi
import threading
import queue

class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Image Recognition")

        # Initialize variables
        self.cap = None  # This will be set after webcam selection
        self.is_running = False
        self.target_image = None
        self.target_image_path = None
        self.available_webcams = self.get_available_webcams()
        self.match_found= False

        self.recognition_queue = queue.Queue()  # Queue for handling recognition results
        self.recognition_thread = None  # Thread for recognition
        self.is_running = False  # Control flag

        # Get the current script's directory
        script_directory = os.path.dirname(os.path.abspath(__file__))
        
        # Set the default image folder relative to the script directory
        self.default_image_folder = os.path.join(script_directory, 'images')  # Folder named 'images'
        
        # Set the current folder to the default image folder
        self.current_image_folder = self.default_image_folder

        # Define the size of the "playing card" area (width, height)
        self.card_width = 200
        self.card_height = 300

        # Coordinates for the ROI (Region of Interest) - where the playing card sized area will be placed
        self.roi_x = 50  # X coordinate for the top-left corner
        self.roi_y = 50  # Y coordinate for the top-left corner

        # Create GUI components
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()

        # Video frame on the left
        self.video_frame = tk.Label(self.main_frame)
        self.video_frame.grid(row=0, column=0)

        # Debug log frame on the right
        self.debug_frame = tk.Frame(self.main_frame)
        self.debug_frame.grid(row=0, column=1, padx=10)

        # Scrollable Text widget for the debug log
        self.debug_log = tk.Text(self.debug_frame, height=20, width=40, wrap=tk.WORD, state=tk.DISABLED)
        self.debug_log.grid(row=0, column=0)

        self.start_button = tk.Button(self.root, text="Start Webcam", command=self.start_webcam)
        self.stop_button = tk.Button(self.root, text="Stop Webcam", command=self.stop_webcam, state=tk.DISABLED)
        self.select_button = tk.Button(self.root, text="Load Image Folder", command=self.select_image_folder)
        self.folder_label = tk.Label(self.root, text=f"Current Folder: {self.current_image_folder}")
        # Webcam selection dropdown
        self.webcam_label = tk.Label(self.root, text="Select Webcam:")
        
        self.webcam_combobox = ttk.Combobox(self.root, values=self.available_webcams)
        
        # Create a frame to hold buttons more compactly
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=5)

        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.select_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Place the folder label and webcam selection next to each other
        self.folder_label.pack(pady=5)
        self.webcam_label.pack(side=tk.LEFT, padx=5)
        self.webcam_combobox.pack(side=tk.LEFT, padx=5)
        
        # Default to first webcam in the list
        if self.available_webcams:
            self.webcam_combobox.set(self.available_webcams[0])

        # Bind mouse events for moving/resizing the ROI
        self.video_frame.grid(row=0, column=0, sticky="nsew")  # Allow expansion
        self.main_frame.columnconfigure(0, weight=1)  # Expand to fill space
        self.main_frame.rowconfigure(0, weight=1)

        self.roi_dragging = False
        self.roi_resizing = False
        self.roi_drag_offset = (0, 0)

        # Slider to control the image recognition frequency
        self.freq_label = tk.Label(self.root, text="Image Recognition Frequency (ms):")
        self.freq_label.pack()

        self.freq_slider = tk.Scale(self.root, from_=10, to_=2000, orient=tk.HORIZONTAL, label="Frequency (ms)")
        self.freq_slider.set(200)  # Default frequency is 150 ms (6 calls per second)
        self.freq_slider.pack()

        # Default value for the frequency (in milliseconds)
        self.recognition_frequency = self.freq_slider.get()

        # Slider to adjust the image detection threshold
        self.threshold_label = tk.Label(self.root, text="Image Detection Threshold:")
        self.threshold_label.pack()

        self.threshold_slider = tk.Scale(self.root, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, label="Threshold")
        self.threshold_slider.set(0.4)  # Default threshold value
        self.threshold_slider.pack()
        
        self.match_label = tk.Label(self.root, text="", font=("Arial", 12, "bold"), fg="green")
        self.match_label.pack()

    def get_available_webcams(self):
        """ Get list of available webcams with descriptive names. """
        available_devices = []
        w = wmi.WMI()

        for i in range(5):  # Check the first 5 devices (adjust if necessary)
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                # Attempt to fetch device description using WMI for more detailed names
                device_name = f"Webcam {i}"
                try:
                    # Using WMI to get video devices and their names
                    for device in w.query("SELECT * FROM Win32_PnPEntity WHERE DeviceID LIKE '%VID%'"):
                        if 'camera' in device.Caption.lower():
                            device_name = device.Caption  # This should give the full device name
                            break
                except Exception as e:
                    device_name = f"Webcam {i} - {str(e)}"
                available_devices.append(f"Webcam {i} - {device_name}")
                cap.release()
        return available_devices

    def start_webcam(self):
        """ Start the selected webcam feed and recognition loop. """
        selected_webcam = self.webcam_combobox.get()
        webcam_index = int(selected_webcam.split(" ")[1])  # Extract webcam index
        
        self.cap = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)  # Use DirectShow backend
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open the selected webcam.")
            return

        # Read a frame to get the dimensions
        ret, frame = self.cap.read()
        if ret:
            frame_height, frame_width, _ = frame.shape

            # Calculate center position for ROI
            self.roi_x = (frame_width - self.card_width) // 2
            self.roi_y = (frame_height - self.card_height) // 2

        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.update_frame()

        # Log message
        self.log_debug_message(f"Started webcam: {selected_webcam} (ROI centered at: {self.roi_x}, {self.roi_y})")

    def stop_webcam(self):
        """ Stop the webcam feed. """
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        self.video_frame.config(image="")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        # Log message
        self.log_debug_message("Webcam stopped.")

    def update_frame(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                color = (0, 0, 255) if self.match_found else (0, 255, 0)
                cv2.rectangle(frame, (self.roi_x, self.roi_y),
                              (self.roi_x + self.card_width, self.roi_y + self.card_height), color, 3)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                # Resize the frame to fit within a specified size (adjust as needed)
                desired_width = 400  # Set a fixed width or make it dynamic
                desired_height = 300  # Adjust height accordingly
                image_resized = image.resize((desired_width, desired_height), Image.LANCZOS)

                photo = ImageTk.PhotoImage(image=image_resized)
                self.video_frame.config(image=photo)
                self.video_frame.image = photo


                # Start recognition in a separate thread if not already running
                if self.recognition_thread is None or not self.recognition_thread.is_alive():
                    self.recognition_thread = threading.Thread(target=self.perform_image_recognition, args=(frame,))
                    self.recognition_thread.daemon = True
                    self.recognition_thread.start()

                # Process results from the queue
                self.process_recognition_results()

                self.root.after(self.recognition_frequency, self.update_frame)

    def process_recognition_results(self):
        """ Safely update UI from the main thread. """
        try:
            while not self.recognition_queue.empty():
                match_found = self.recognition_queue.get_nowait()
                self.match_found = match_found
                self.match_label.config(text="Match Found!" if match_found else "")
                if match_found:
                    self.log_debug_message("Image match detected!")
                    self.root.after(1000, self.clear_match_label)
        except queue.Empty:
            pass

    def select_image_folder(self):
        """ Let the user select a folder of images. """
        folder_path = filedialog.askdirectory(initialdir=self.current_image_folder)
        if folder_path:
            self.current_image_folder = folder_path
            self.folder_label.config(text=f"Current Folder: {self.current_image_folder}")
            image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
            if image_files:
                self.target_image_path = os.path.join(folder_path, image_files[0])
                self.target_image = cv2.imread(self.target_image_path, cv2.IMREAD_GRAYSCALE)
                messagebox.showinfo("Image Loaded", f"Loaded image: {self.target_image_path}")
                self.log_debug_message(f"Loaded image: {self.target_image_path}")
            else:
                messagebox.showerror("Error", "No PNG images found in the selected folder.")

    def perform_image_recognition(self, frame):
        """ Perform image recognition in a separate thread. """
        if self.target_image is not None:
            roi_frame = frame[self.roi_y:self.roi_y + self.card_height, self.roi_x:self.roi_x + self.card_width]
            gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

            res = cv2.matchTemplate(gray_frame, self.target_image, cv2.TM_CCOEFF_NORMED)
            threshold = self.threshold_slider.get()
            loc = np.where(res >= threshold)

            match_found = len(loc[0]) > 0
            self.recognition_queue.put(match_found)  # Send result to the queue

    def clear_match_label(self):
        self.match_label.config(text="")

    def log_debug_message(self, message):
        """ Log debug messages to the Text widget. """
        self.debug_log.config(state=tk.NORMAL)  # Enable text widget for editing
        self.debug_log.insert(tk.END, message + "\n")  # Insert the message at the end
        self.debug_log.yview(tk.END)  # Scroll to the end to show the latest message
        self.debug_log.config(state=tk.DISABLED)  # Disable text widget to prevent manual editing

    def export_to_obs(self, image_path):
        """ Export the matched image to OBS as an image source. """
        # OBS integration (if needed) can go here

    def on_roi_press(self, event):
        """ Capture the starting point of the ROI move or resize action. """
        # Check if user clicked inside the ROI for moving or resizing
        if self.roi_x <= event.x <= self.roi_x + self.card_width and self.roi_y <= event.y <= self.roi_y + self.card_height:
            self.roi_dragging = True
            self.roi_drag_offset = (event.x - self.roi_x, event.y - self.roi_y)

    def on_roi_drag(self, event):
        """ Update the position of the ROI during dragging. """
        if self.roi_dragging:
            self.roi_x = event.x - self.roi_drag_offset[0]
            self.roi_y = event.y - self.roi_drag_offset[1]

    def on_roi_release(self, event):
        """ End the ROI dragging action. """
        self.roi_dragging = False


if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    root.mainloop()
