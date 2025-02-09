import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
from PIL import Image, ImageTk
import threading
import queue
import random
from pygrabber.dshow_graph import FilterGraph
import time
import concurrent.futures
import numpy as np

class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Image Recognition")

        self.dragging_point = None # Stores which point of the polygon is being dragged

        # Initialize polygon with 4 points (modify as needed)
        self.polygon = np.array([
            [100, 100],  # Top-left
            [200, 100],  # Top-right
            [200, 200],  # Bottom-right
            [100, 200]   # Bottom-left
        ], dtype=np.int32)

        cv2.namedWindow("Webcam")
        cv2.setMouseCallback("Webcam", self.mouse_callback)
        
        #choose from rectangle, polygon, or auto - to do: make this selectable from a drop down, default to polygon (or from a config file's default)
        self.detect_mode = "rectangle"

        # Initialize variables
        self.cap = None  # This will be set after webcam selection
        self.is_running = False
        self.match_found = False
        self.recognition_queue = queue.Queue() # Queue for handling recognition results
        self.recognition_thread = None
        self.target_images = []
        self.matched_image_path = None
        self.available_webcams = self.find_webcams()

        # Get the current script's directory
        script_directory = os.path.dirname(os.path.abspath(__file__))
        
        # Set the default image folders relative to the script directory - low resolution images for parsing/matching,
        # and high resolution files for displaying once a match is found.
        # IMPORTANT: The file names in these folders must match exactly. I have found Windows PowerToys Image Resizer to be helpful for scaling files:
        # https://learn.microsoft.com/en-us/windows/powertoys/install
        self.default_image_folder = os.path.join(script_directory, 'low_res_images')  # Folder named 'images'
        self.high_res_image_folder = os.path.join(script_directory, 'high_res_images')  # Folder named 'images'
        # Set the current folder to the default image folder
        self.low_res_image_folder = self.default_image_folder

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

        self.video_frame.bind("<ButtonPress-1>", self.on_roi_press)
        self.video_frame.bind("<B1-Motion>", self.on_roi_drag)
        self.video_frame.bind("<ButtonRelease-1>", self.on_roi_release)
        self.video_frame.bind("<Motion>", self.on_mouse_move)

        # Debug log frame on the right
        self.debug_frame = tk.Frame(self.main_frame)
        self.debug_frame.grid(row=0, column=1, padx=10)
                
        self.match_frame = tk.Label(self.main_frame)
        self.match_frame.grid(row=0, column=2, padx=10)

        # Scrollable Text widget for the debug log
        self.debug_log = tk.Text(self.debug_frame, height=20, width=100, wrap=tk.WORD, state=tk.DISABLED)
        self.debug_log.grid(row=0, column=0)

        self.start_button = tk.Button(self.root, text="Start Webcam", command=self.start_webcam)
        self.stop_button = tk.Button(self.root, text="Stop Webcam", command=self.stop_webcam, state=tk.DISABLED)
        self.select_button = tk.Button(self.root, text="Load Image Folder", command=self.select_image_folder)
        self.folder_label = tk.Label(self.root, text=f"Current Folder: {self.low_res_image_folder}")
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
        if len(self.available_webcams) >= 2:
            self.webcam_combobox.set(self.available_webcams[1]) #default to #1 for Eric's Machine
        elif self.available_webcams:
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

        self.freq_slider = tk.Scale(self.root, from_=10, to_=2000, orient=tk.HORIZONTAL, label="Frequency (ms)", command=self.update_frequency)
        self.freq_slider.set(100)  # Default frequency is 100 ms (10 calls per second)
        self.freq_slider.pack()

        # Default value for the frequency (in milliseconds)
        self.recognition_frequency = self.freq_slider.get()

        # Slider to adjust the image detection threshold
        self.threshold_label = tk.Label(self.root, text="Image Detection Threshold (perc of keypoints:")
        self.threshold_label.pack()

        self.threshold_slider = tk.Scale(self.root, from_=0, to=100, resolution=1, orient=tk.HORIZONTAL, label="Threshold", command=self.update_threshold)
        self.threshold_slider.set(20)  # Default threshold
        self.threshold_slider.pack()
        self.match_threshold = self.threshold_slider.get()  # Set initial match threshold in case it isn't used.
        
        self.match_label = tk.Label(self.root, text="", font=("Arial", 12, "bold"), fg="green")
        self.match_label.pack()

        self.video_width = 640  # Default width, update dynamically if needed
        self.video_height = 480  # Default height, update dynamically if needed

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.worker_threads = 4 # todo: make configurable in UI

        #create image recognition objects for repeated use
        self.orb = cv2.ORB_create()
        # FLANN Matcher Parameters (optimized for ORB/SIFT)
        index_params = dict(algorithm=6,  # FLANN LSH (Locality Sensitive Hashing) for ORB
                            table_number=6,  # Number of hash tables
                            key_size=12,  # Size of the key in bits
                            multi_probe_level=1)  # Number of probes per table

        search_params = dict(checks=50)  # Number of nearest neighbors to check

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)


    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Mouse click
            for i, (px, py) in enumerate(self.polygon):
                if abs(x - px) < 10 and abs(y - py) < 10:  # Click near a point
                    self.dragging_point = i  # Store index of point
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_point is not None:
            self.polygon[self.dragging_point] = [x, y]  # Move point
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_point = None  # Stop dragging

    def find_webcams(self):
        """Find available webcams and get their descriptive names."""
        webcams = []
        graph = FilterGraph()
        devices = graph.get_input_devices()  # List of webcam names
        
        for index, device_name in enumerate(devices):
            webcams.append((index, device_name))  # Store index and name as a tuple
        
        return webcams

    def start_webcam(self):
        """ Start the selected webcam feed and recognition loop. """
        selected_webcam = self.webcam_combobox.get()
        webcam_index = int(selected_webcam.split(" ")[0])  # Extract webcam index
        
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
            self.cap = None
        self.video_frame.config(image="")
        self.match_frame.config(image="")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        # Log message
        self.log_debug_message("Webcam stopped.")

    def on_closing(self):
        self.stop_webcam()
        self.root.destroy()

    def update_frequency(self, value):
        self.recognition_frequency = int(value)

    def update_threshold(self, value):
        self.match_threshold = float(value)
        self.log_debug_message(f"Updated match Threshold to {self.match_threshold}")

    def update_frame(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:

                self.draw_roi_frame(frame)
                #self.log_debug_message("in ret loop")

                # Start recognition in a separate thread if not already running
                if self.recognition_thread is None or not self.recognition_thread.is_alive():
                    #self.log_debug_message("alive!")
                    self.recognition_thread = threading.Thread(target=self.perform_image_recognition, args=(frame,))
                    self.recognition_thread.daemon = True
                    self.recognition_thread.start()

                # Process results from the queue
                self.process_recognition_results()

                self.root.after(self.recognition_frequency, self.update_frame)

    def draw_roi_frame(self, frame):
        if self.detect_mode == "rectangle":
            color = (0, 0, 255) if self.match_found else (0, 255, 0)
            cv2.rectangle(frame, (self.roi_x, self.roi_y),
                            (self.roi_x + self.card_width, self.roi_y + self.card_height), color, 3)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Resize the frame to fit within a specified size (adjust as needed)
            image_resized = image.resize((self.video_width, self.video_height), Image.LANCZOS)

            photo = ImageTk.PhotoImage(image=image_resized)
            self.video_frame.config(image=photo)
            self.video_frame.image = photo
        elif self.detect_mode == "polygon":
            pass #to do: create polygon detect mode
        elif self.detect_mode == "auto":
            pass #to do: create automatic detect mode

    def process_recognition_results(self):
        """ Safely update UI from the main thread. """
        try:
            while not self.recognition_queue.empty():
                match_found = self.recognition_queue.get_nowait()

                if match_found:
                    self.matched_image_path = match_found
                    self.match_label.config(text=f"Matched {self.matched_image_path}")
                    self.display_matched_image()
                    self.log_debug_message(f"Image match detected - {self.matched_image_path}")
                    self.root.after(3000, self.clear_match_label)
        except queue.Empty:
            pass

    def clear_match_label(self):
        self.match_label.config(text="")

    def display_matched_image(self):
        if self.matched_image_path:
            image = Image.open(self.matched_image_path)
            image_resized = image.resize((300, 419), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image=image_resized)
            self.match_frame.config(image=photo)
            self.match_frame.image = photo

            # Save the high-res matched image to a file for OBS to import
            export_path = os.path.join(os.path.dirname(__file__), "obs_export_image.png")
            image.save(export_path)
            self.log_debug_message(f"Saved high-res matched image to {export_path}")

    def select_image_folder(self):
        """ Let the user select a folder of images. """
        folder_path = filedialog.askdirectory(initialdir=self.low_res_image_folder)
        if folder_path:
            start_time = time.time()
            self.low_res_image_folder = folder_path
            self.folder_label.config(text=f"Current Folder: {self.low_res_image_folder}")
            self.target_images.clear()

            def process_image(path):
                target_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                rotated_stats = []
                for angle in [0, 90, 180, 270]:
                    rotated_target = self.rotate_image(target_image, angle)
                    rotated_stats.append(self.orb.detectAndCompute(rotated_target, None))
                return rotated_stats

            paths = os.listdir(folder_path)

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.worker_threads) as executor:
                future2path = {}
                for image_path in paths:
                    if image_path.endswith('.png') or image_path.endswith('.jpg'):
                        full_path = os.path.join(self.low_res_image_folder, image_path)
                        future2path[executor.submit(process_image, full_path)] = image_path

                for future in concurrent.futures.as_completed(future2path):
                    image_path = future2path[future]
                    try:
                        image_stats = future.result()
                    except Exception as exc:
                        print(f'{image_path} generated an exception: {exc}')
                    else:
                        self.target_images.append((image_path, image_stats))

            if not self.target_images:
                messagebox.showerror("Error", "No PNG or JPG images found in the selected folder.")

            end_time = time.time()
            elapsed_time = end_time - start_time
            self.log_debug_message(f"Loaded {len(self.target_images)} images in {elapsed_time:.4f}s.")
            #randomize the order to remove selection bias
            random.shuffle(self.target_images)

    def draw_and_pause(self, image1, keypoints1, image2, keypoints2, matches):
        """ Draws matches and pauses execution until a key is pressed """
        matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Show the matched features
        cv2.imshow("Matches", matched_image)

        # Wait for user input to continue (press any key)
        print("Press any key to continue to the next image...")
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()

    def draw(self, image1, keypoints1, image2, keypoints2, matches):
        """ Draws matches and pauses execution until a key is pressed """

        #destroy the last window
        cv2.destroyAllWindows()

        matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Show the matched features
        cv2.imshow("Matches", matched_image)

        # Wait for user input to continue (press any key)

    def rotate_image(self, image, angle):
        """ Rotate image by a specified angle. """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        return rotated

    def perform_image_recognition(self, frame):
        """ Perform image recognition in a separate thread. """
        start_time = time.time()
        knn_time = 0
        filter_time = 0
        if self.low_res_image_folder and self.target_images:

            roi_frame = frame[self.roi_y:self.roi_y + self.card_height, self.roi_x:self.roi_x + self.card_width]
            #gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

            #cv2.imshow("ROI Frame", roi_frame)
            #cv2.waitKey(1) #Ensures the OpenCV window refreshes

            width = 95
            height = 133

            roi_frame = cv2.resize(roi_frame, (width, height))
            
            kp2, des2 = self.orb.detectAndCompute(roi_frame, None)
            
            # create BFMatcher object
            #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

            #before we look through any images, reset our scores to zero
            match_path = None
            lowest_dist = 300.0 #use a high number to start - best matches are the lowest distance

            def process_stat(stat, d2):
                inner_dist = lowest_dist  # use a high number to start - best matches are the lowest distance
                knn_t = 0
                filter_t = 0

                for kp1, des1 in stat:
                    if des1 is None or des2 is None:
                        continue  # Avoid running knnMatch() on None values
                    probe_start = time.time()
                    matches = self.flann.knnMatch(des1, d2, k=min(2, len(d2)))
                    probe_end = time.time()
                    knn_t += probe_end - probe_start

                    # # Apply Lowe's ratio test (helps remove false matches)
                    probe_start = time.time()
                    for match in matches:
                        if len(match) < 2:
                            continue  # skip if there aren't at least two matches
                        m, n = match[:2]  # Unpack only the first two matches

                        # check to see if m is significantly better than n, and if so, consider it a good match
                        # the lower the threshold, the stricter the test
                        if m.distance < 0.75 * n.distance:  # adjust ratio as needed
                            if m.distance < inner_dist:
                                # set the new best score (smallest distance)
                                inner_dist = m.distance

                    probe_end = time.time()
                    filter_t += probe_end - probe_start
                return inner_dist, knn_t, filter_t

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.worker_threads) as executor:
                future2path = {}
                for (image, stats) in self.target_images:
                    future2path[executor.submit(process_stat, stats, des2)] = image

                for future in concurrent.futures.as_completed(future2path):
                    image_path = future2path[future]
                    try:
                        dist, knn, f = future.result()
                    except Exception as exc:
                        print(f'{image_path} generated an exception: {exc}')
                    else:
                        knn_time += knn
                        filter_time += f
                        if dist < lowest_dist:
                            lowest_dist = dist
                            match_path = image_path

            if match_path and lowest_dist < self.match_threshold:
                if match_path.strip().startswith("alt_"):
                    match_path = match_path[4:]

                #self.draw_and_pause(target_image, kp1, roi_frame, kp2, match)
                high_res_path = os.path.join(self.high_res_image_folder, match_path)
                self.log_debug_message(f"Match detected (distance of {lowest_dist}) - adding {match_path} to recognition_queue!")
                # Ensure the high-resolution file exists before adding it to the queue
                if os.path.exists(high_res_path):
                    self.log_debug_message(f"Match detected - using high-res version: {high_res_path}")
                    self.recognition_queue.put(high_res_path)
                else:
                    low_res_path = os.path.join(self.low_res_image_folder, match_path)
                    if os.path.exists(high_res_path):
                        self.log_debug_message(f"High-resolution version not found ({high_res_path}), using low-res: {low_res_path}")
                        self.recognition_queue.put(low_res_path)
                    else:
                        self.log_debug_message(f"No path found")


            #print how long parsing all of the images took
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.log_debug_message(f"Image recognition took {elapsed_time:.4f} seconds for {len(self.target_images)} images.")
            self.log_debug_message(f"knn: {knn_time:.4f}, filter: {filter_time:.4f}")


    def log_debug_message(self, message):
        """ Log debug messages to the Text widget. """
        self.debug_log.config(state=tk.NORMAL)  # Enable text widget for editing
        self.debug_log.insert(tk.END, message + "\n")  # Insert the message at the end
        self.debug_log.yview(tk.END)  # Scroll to the end to show the latest message
        self.debug_log.config(state=tk.DISABLED)  # Disable text widget to prevent manual editing

    def export_to_obs(self, image_path):
        """ Export the matched image to OBS as an image source. """ 
        # OBS integration (if needed) can go here

    # def on_roi_press(self, event):
    #     """ Capture the starting point of the ROI move or resize action. """
    #     # Check if user clicked inside the ROI for moving or resizing
    #     if self.roi_x <= event.x <= self.roi_x + self.card_width and self.roi_y <= event.y <= self.roi_y + self.card_height:
    #         self.roi_dragging = True
    #         self.roi_drag_offset = (event.x - self.roi_x, event.y - self.roi_y)

    def on_roi_drag(self, event):
        if self.roi_dragging:
            self.roi_x = max(0, min(event.x - self.roi_drag_offset[0], int(self.cap.get(3)) - self.card_width))
            self.roi_y = max(0, min(event.y - self.roi_drag_offset[1], int(self.cap.get(4)) - self.card_height))


    def on_roi_release(self, event):
        """ End the ROI dragging action. """
        self.roi_dragging = False

    def on_roi_press(self, event):
        """Handle mouse button press on ROI."""
        x, y = event.x, event.y
        margin = 10  # Sensitivity for resizing

        # Detect which part of ROI is clicked (corners for resizing, inside for dragging)
        if (self.roi_x - margin <= x <= self.roi_x + margin and
            self.roi_y - margin <= y <= self.roi_y + margin):
            self.roi_resizing = "top_left"
        elif (self.roi_x + self.card_width - margin <= x <= self.roi_x + self.card_width + margin and
            self.roi_y - margin <= y <= self.roi_y + margin):
            self.roi_resizing = "top_right"
        elif (self.roi_x - margin <= x <= self.roi_x + margin and
            self.roi_y + self.card_height - margin <= y <= self.roi_y + self.card_height + margin):
            self.roi_resizing = "bottom_left"
        elif (self.roi_x + self.card_width - margin <= x <= self.roi_x + self.card_width + margin and
            self.roi_y + self.card_height - margin <= y <= self.roi_y + self.card_height + margin):
            self.roi_resizing = "bottom_right"
        elif (self.roi_x <= x <= self.roi_x + self.card_width and
            self.roi_y <= y <= self.roi_y + self.card_height):
            self.roi_dragging = True
            self.roi_drag_offset = (x - self.roi_x, y - self.roi_y)
        else:
            self.roi_resizing = None
            self.roi_dragging = False

    # def on_roi_drag(self, event):
    #     """Handle dragging or resizing of the ROI, optimized for performance."""
    #     x, y = event.x, event.y
    #     changed = False  # Flag to track if the ROI has changed

    #     if self.roi_dragging:
    #         new_x = x - self.roi_drag_offset[0]
    #         new_y = y - self.roi_drag_offset[1]

    #         # Prevent ROI from moving off-screen
    #         if 0 <= new_x <= self.video_width - self.card_width:
    #             self.roi_x = new_x
    #             changed = True
    #         if 0 <= new_y <= self.video_height - self.card_height:
    #             self.roi_y = new_y
    #             changed = True

    #     elif self.roi_resizing:
    #         min_size = 50  # Prevent ROI from being too small

    #         if self.roi_resizing == "top_left":
    #             new_width = self.card_width + (self.roi_x - x)
    #             new_height = self.card_height + (self.roi_y - y)
    #             if new_width >= min_size and 0 <= x < self.roi_x + self.card_width:
    #                 self.roi_x = x
    #                 self.card_width = new_width
    #                 changed = True
    #             if new_height >= min_size and 0 <= y < self.roi_y + self.card_height:
    #                 self.roi_y = y
    #                 self.card_height = new_height
    #                 changed = True

    #         elif self.roi_resizing == "top_right":
    #             new_width = x - self.roi_x
    #             new_height = self.card_height + (self.roi_y - y)
    #             if new_width >= min_size and x <= self.video_width:
    #                 self.card_width = new_width
    #                 changed = True
    #             if new_height >= min_size and 0 <= y < self.roi_y + self.card_height:
    #                 self.roi_y = y
    #                 self.card_height = new_height
    #                 changed = True

    #         elif self.roi_resizing == "bottom_left":
    #             new_width = self.card_width + (self.roi_x - x)
    #             new_height = y - self.roi_y
    #             if new_width >= min_size and 0 <= x < self.roi_x + self.card_width:
    #                 self.roi_x = x
    #                 self.card_width = new_width
    #                 changed = True
    #             if new_height >= min_size and y <= self.video_height:
    #                 self.card_height = new_height
    #                 changed = True

    #         elif self.roi_resizing == "bottom_right":
    #             new_width = x - self.roi_x
    #             new_height = y - self.roi_y
    #             if new_width >= min_size and x <= self.video_width:
    #                 self.card_width = new_width
    #                 changed = True
    #             if new_height >= min_size and y <= self.video_height:
    #                 self.card_height = new_height
    #                 changed = True

    #     if changed:
    #         self.update_frame()  # Only update frame if something changed

    # def on_roi_release(self, event):
    #     """Reset dragging and resizing flags."""
    #     self.roi_dragging = False
    #     self.roi_resizing = None

    def on_mouse_move(self, event):
        """Change cursor when near ROI edges to indicate resizing."""
        x, y = event.x, event.y
        margin = 10

        if (self.roi_x - margin <= x <= self.roi_x + margin and
            self.roi_y - margin <= y <= self.roi_y + margin):
            self.video_frame.config(cursor="size_nw_se")
        elif (self.roi_x + self.card_width - margin <= x <= self.roi_x + self.card_width + margin and
            self.roi_y - margin <= y <= self.roi_y + margin):
            self.video_frame.config(cursor="size_ne_sw")
        elif (self.roi_x - margin <= x <= self.roi_x + margin and
            self.roi_y + self.card_height - margin <= y <= self.roi_y + self.card_height + margin):
            self.video_frame.config(cursor="size_ne_sw")
        elif (self.roi_x + self.card_width - margin <= x <= self.roi_x + self.card_width + margin and
            self.roi_y + self.card_height - margin <= y <= self.roi_y + self.card_height + margin):
            self.video_frame.config(cursor="size_nw_se")
        elif (self.roi_x <= x <= self.roi_x + self.card_width and
            self.roi_y <= y <= self.roi_y + self.card_height):
            self.video_frame.config(cursor="fleur")
        else:
            self.video_frame.config(cursor="")


if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    root.mainloop()
