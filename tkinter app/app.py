import cv2
import os 
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox,filedialog
from collections import deque
from PIL import Image, ImageTk,ImageGrab
from rembg import remove

class KalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
        self.kalman.statePost = np.zeros((4, 1), dtype=np.float32)

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]], dtype=np.float32)
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        return prediction[0, 0], prediction[1, 0]

# Initialize Kalman filters for left and right earrings
kalman_left = KalmanFilter()
kalman_right = KalmanFilter()


class VirtualTryOnApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vashions")
        self.root.iconbitmap('static/vashions_icon.ico')  # Replace 'path_to_icon.ico' with the path to your .ico file
        self.current_design = None
        self.create_widgets()
        self.face_bbox_buffer = deque(maxlen=10)
        self.hand_positions = deque(maxlen=5)
        self.upload_counter = 0

    def create_widgets(self):
        # Create a main frame to hold the left and right frames
            self.main_frame = tk.Frame(self.root, width=1290, height=720)
            self.main_frame.pack()

            # Create the left frame
            self.left_frame = tk.Frame(self.main_frame, width=400, height=720, bg="black")
            self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
            self.left_frame.pack_propagate(False)  # Prevent resizing

            # Heading
            self.heading = ttk.Label(self.left_frame, text="Virtual Try On", font=("Helvetica", 24),background="black", foreground="lightgray")
            self.heading.pack(pady=20)

            # Necklace button
            self.necklace_button = ttk.Button(self.left_frame, text="Necklace", command=lambda: self.show_necklaces("Necklace"))
            self.necklace_button.pack(pady=10)

            # Ring button
            self.ring_button = ttk.Button(self.left_frame, text="Ring", command=lambda: self.show_rings("Ring"))
            self.ring_button.pack(pady=10)

            # Earring button
            self.earring_button = ttk.Button(self.left_frame, text="Earring", command=lambda: self.show_earring("Earring"))
            self.earring_button.pack(pady=10)

            # # Bangle button
            # self.bangle_button = ttk.Button(self.left_frame, text="Bangle", command=lambda: self.show_bangle("Bangle"))
            # self.bangle_button.pack(pady=10)

            # Create the right frame
            self.right_frame = tk.Frame(self.main_frame, width=890, height=720, bg="white")
            self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

            # Load the image
            image_path = 'static/profile.jpg'
            image = Image.open(image_path)
            resized_image = image.resize((890, 720))  # Resize the image to fit the frame
            photo = ImageTk.PhotoImage(resized_image)

            # Create a label with the image and add it to the right frame
            label = tk.Label(self.right_frame, image=photo)
            label.image = photo  # Keep a reference to prevent garbage collection
            label.pack(fill=tk.BOTH, expand=True)

            # Webcam display area on the right frame
            self.webcam_canvas = tk.Canvas(self.right_frame, width=890, height=720, bg="white")
            self.webcam_canvas.pack(fill=tk.BOTH, expand=True)

            # Initialize MediaPipe Face Detection.
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

            # Initialize MediaPipe Hands.
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

            # Initialize MediaPipe Face Mesh
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh()

            # Default necklace path
            self.necklace_image_path = 'static/Image/Necklace/necklace_1.png'
            self.necklace_image = cv2.imread(self.necklace_image_path, cv2.IMREAD_UNCHANGED)
            
            # Default ring path
            self.ring_image_path = 'static/Image/Ring/ring_1.png'
            self.ring_image = cv2.imread(self.ring_image_path, cv2.IMREAD_UNCHANGED)
            
            # Default earring image path
            self.earring_image_path = 'static/Image/Earring/earring_1.png'
            self.earring_image = cv2.imread(self.earring_image_path, cv2.IMREAD_UNCHANGED)

            # Default bangle image path
            self.bangle_image_path = 'static/Image/Bangle/bangle_1.png'
            self.bangle_image = cv2.imread(self.bangle_image_path, cv2.IMREAD_UNCHANGED)
    def process_necklace_image(self, image, output_image_path):
       # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to create a binary image
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

        # Crop the image to the bounding box of the necklace
        x, y, w, h = cv2.boundingRect(mask)
        cropped_result = image[y:y+h, x:x+w]

        cropped_gray = cv2.cvtColor(cropped_result, cv2.COLOR_BGR2GRAY)
        _,  cropped_binary = cv2.threshold(cropped_gray, 1, 255, cv2.THRESH_BINARY)
        cropped_contours, _ = cv2.findContours(cropped_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(cropped_gray)
        cv2.drawContours(mask, cropped_contours, -1, (255), thickness=cv2.FILLED)
        inner_mask = cv2.bitwise_not(mask)

        # Apply the inner mask to create a black background inside the necklace
        bg_inner = np.ones_like(cropped_result, np.uint8) * 0
        bg_inner[inner_mask == 0] = 0
        final_result = cv2.add(cropped_result, bg_inner)

        # Save the cropped image temporarily
        temp_cropped_path = "temp_cropped_image.png"
        cv2.imwrite(temp_cropped_path, final_result)

        # Perform background removal using rembg
        cropped_image_pil = Image.open(temp_cropped_path)
        output_image_pil = remove(cropped_image_pil)
        
        # Save the processed image with high quality
        output_image_pil.save(output_image_path, format="PNG")
        print(f"Processed image saved to '{output_image_path}'")
    def upload_and_process_image(self,design):
        input_image_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
        )
        
        if not input_image_path:
            messagebox.showinfo("No file selected", "Please select an image file to process.")
            return

        image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            messagebox.showerror("Error", f"Failed to load image from '{input_image_path}'. Check the file path and integrity.")
            return

        # Increment the upload counter
        self.upload_counter += 1
        output_image_path = os.path.join("static", "Image", f"{design}", f"saved_image_{self.upload_counter}.png")

        try:
            self.process_necklace_image(image, output_image_path)
            messagebox.showinfo("Success", f"Processed image saved to '{output_image_path}'")
        except Exception as e:
            messagebox.showerror("Error", f"Error processing the image: {e}")


    def show_necklaces(self, design):
        self.current_design = design
        self.necklace_type = 'regular'
        necklace_list = [
            {'path': r"static/Image/Necklace/necklace_1.png", 'type': 'regular'},
            {'path': r"static/Image/Necklace/jewellery_1.png", 'type': 'large'},
            {'path': r"static/Image/Necklace/jewellery_2.png", 'type': 'large2'},
            {'path': r"static/Image/Necklace/jewellery_3.png", 'type': 'regular'},
            {'path': r"static/Image/Necklace/jewellery_4.png", 'type': 'choker'},
            {'path': r"static/Image/Necklace/jewellery_6.png", 'type': 'large2'}
        ]
        self.show_virtual_try_on(necklace_list, design)

    def show_rings(self, design):
        self.current_design = design
        ring_list = [
            {'path': r"static/Image/Ring/ring_1.png", 'type': 'regular'},
            {'path': r"static/Image/Ring/ring_3.png", 'type': 'regular'}, 
            {'path': r"static/Image/Ring/ring_4.png", 'type': 'regular'},
            {'path': r"static/Image/Ring/ring_5.png", 'type': 'regular'}
        ]
        self.show_virtual_try_on(ring_list, design)

    def show_earring(self, design):
        self.current_design = design
        earring_list = [
            {'path': r"static/Image/Earring/earring_1.png", 'type': 'regular'},
            {'path': r"static/Image/Earring/earring_2.png", 'type': 'regular'},
            {'path': r"static/Image/Earring/earring_3.png", 'type': 'regular'},
            {'path': r"static/Image/Earring/earring_4.png", 'type': 'regular'},
            {'path': r"static/Image/Earring/earring_10.png", 'type': 'regular'},
            {'path': r"static/Image/Earring/earring_5.png", 'type': 'regular'}
        ]
        self.show_virtual_try_on(earring_list, design)

    
    def show_bangle(self, design):
        self.current_design = design
        bangle_list = [
            {'path': r"static/Image/Bangle/bangle_1.png", 'type': 'regular'},
            {'path': r"static/Image/Bangle/bangle_2.png", 'type': 'regular'},
            {'path': r"static/Image/Bangle/bangle_3.png", 'type': 'regular'},
            {'path': r"static/Image/Bangle/bangle_4.png", 'type': 'regular'}
        ]
        self.show_virtual_try_on(bangle_list, design)


    def change_necklace_image(self, path):
        self.necklace_image_path = path
        self.necklace_image = cv2.imread(self.necklace_image_path, cv2.IMREAD_UNCHANGED)
        print(f"Changed necklace image to {self.necklace_image_path}")

    def change_ring_image(self, path):
        self.ring_image_path = path
        self.ring_image = cv2.imread(self.ring_image_path, cv2.IMREAD_UNCHANGED)
        print(f"Changed ring image to {self.ring_image_path}")

    def change_earring_image(self, path):
        self.earring_image_path = path
        self.earring_image = cv2.imread(self.earring_image_path, cv2.IMREAD_UNCHANGED)
        print(f"Changed earring image to {self.earring_image_path}")

    def change_bangle_image(self, path):
        self.bangle_image_path = path
        self.bangle_image = cv2.imread(self.bangle_image_path, cv2.IMREAD_UNCHANGED)
        print(f"Changed bangle image to {self.bangle_image_path}")


    def show_virtual_try_on(self, image_list, design):
        self.try_on_window = tk.Toplevel(self.root)
        self.try_on_window.title("Virtual Try On")
        self.try_on_window.is_fullscreen = False
        main_frame = tk.Frame(self.try_on_window)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.webcam_canvas = tk.Canvas(left_frame, width=650, height=650)
        self.webcam_canvas.pack(side=tk.TOP, pady=10)

        button_frame = tk.Frame(left_frame)
        button_frame.pack(side=tk.TOP, pady=10)

        self.quit_icon = ImageTk.PhotoImage(Image.open("static/icons/quit_icon.png").resize((20, 20), Image.LANCZOS))
        self.capture_icon = ImageTk.PhotoImage(Image.open("static/icons/capture_icon.png").resize((20, 20), Image.LANCZOS))
        self.fullscreen_icon = ImageTk.PhotoImage(Image.open("static/icons/fullscreen_icon.png").resize((20, 20), Image.LANCZOS))

        button1 = tk.Button(button_frame, command=self.button1_action, image=self.quit_icon)
        button1.pack(side=tk.LEFT)

        button3 = tk.Button(button_frame, command=self.button3_action, image=self.capture_icon)
        button3.pack(side=tk.LEFT)

        button2 = tk.Button(button_frame, command=self.button2_action, image=self.fullscreen_icon)
        button2.pack(side=tk.LEFT)

        # self.upload_button = tk.Button(button_frame, text="Upload Image", command=lambda: self.upload_and_process_image(design))
        # self.upload_button.pack(side=tk.LEFT)

        scrollbar = tk.Scrollbar(right_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas = tk.Canvas(right_frame, yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=canvas.yview)

        image_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=image_frame, anchor=tk.NW)

        def update_scroll_region(event):
            canvas.config(scrollregion=canvas.bbox(tk.ALL))

        image_frame.bind("<Configure>", update_scroll_region)

        for idx, image_data in enumerate(image_list):
            row = idx // 2
            col = idx % 2

            image = Image.open(image_data['path'])
            image = image.resize((150, 150), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)

            image_label = tk.Label(image_frame, image=photo)
            image_label.image = photo
            image_label.image_data = image_data
            image_label.grid(row=row * 2, column=col, pady=(10, 0), padx=10)

            item_label = tk.Label(image_frame, text=f"{design} {idx + 1} ({image_data['type']})")
            item_label.grid(row=row * 2 + 1, column=col, pady=(0, 10), padx=10)
            image_label.bind("<Button-1>", self.on_image_click)

        self.cap = cv2.VideoCapture(0)
        self.update_frame()


    def button1_action(self):
        self.cap.release()
        self.try_on_window.destroy()

    def button2_action(self):
        self.try_on_window.is_fullscreen = not self.try_on_window.is_fullscreen
        self.try_on_window.attributes("-fullscreen", self.try_on_window.is_fullscreen)

    def button3_action(self):
        self.root.update_idletasks()
        webcam_feed_width = 760
        webcam_feed_height = 800
        canvas_x = self.webcam_canvas.winfo_rootx()
        canvas_y = self.webcam_canvas.winfo_rooty()
        x1 = canvas_x + (self.webcam_canvas.winfo_width() - webcam_feed_width + 175)
        y1 = canvas_y + (self.webcam_canvas.winfo_height() + 167 - webcam_feed_height)
        x2 = x1 + webcam_feed_width
        y2 = y1 + webcam_feed_height
        image = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        image.save("virtual_try_on_capture.png")

    def rotate_image(self, image, angle):
                    height, width = image.shape[:2]
                    diagonal = int(np.sqrt(height**2 + width**2))
                    
                    padded_image = np.zeros((diagonal, diagonal, image.shape[2]), dtype=np.uint8)
                    x_offset = (diagonal - width) // 2
                    y_offset = (diagonal - height) // 2
                    
                    padded_image[y_offset:y_offset + height, x_offset:x_offset + width] = image
                    
                    center = (diagonal // 2, diagonal // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
                    rotated_image = cv2.warpAffine(padded_image, rotation_matrix, (diagonal, diagonal), 
                                                flags=cv2.INTER_LINEAR)
                    
                    return rotated_image

    def button3_action(self):
        # Ensure the canvas is fully visible and updated before capturing
        # Ensure the canvas is fully visible and updated before capturing
        self.root.update_idletasks()
        
        # Define the exact dimensions of the webcam feed area within the canvas
        webcam_feed_width = 760 # Set this to the width of your webcam feed
        webcam_feed_height = 800  # Set this to the height of your webcam feed

        # Calculate the coordinates of the webcam feed area within the canvas
        canvas_x = self.webcam_canvas.winfo_rootx()
        canvas_y = self.webcam_canvas.winfo_rooty()

        # Calculate the bounding box for the webcam feed area
        x1 = canvas_x + (self.webcam_canvas.winfo_width() - webcam_feed_width + 175)
        y1 = canvas_y + (self.webcam_canvas.winfo_height() + 167 - webcam_feed_height)
        x2 = x1 + webcam_feed_width
        y2 = y1 + webcam_feed_height

        # Capture the image of the webcam feed area within the canvas
        image = ImageGrab.grab(bbox=(x1, y1, x2, y2))

        # Save the captured image
        image.save("virtual_try_on_capture.png")

        print("Virtual try-on image captured and saved as 'virtual_try_on_capture.png'")
        
    def on_image_click(self, event):
        if hasattr(event.widget, 'image_data'):
            image_data = event.widget.image_data
            if self.current_design == "Necklace":
                self.necklace_type = image_data['type']
                self.change_necklace_image(image_data['path'])
            elif self.current_design == "Ring":
                self.change_ring_image(image_data['path'])
            elif self.current_design == "Earring":
                self.change_earring_image(image_data['path'])
            elif self.current_design == "Bangle":
                self.change_bangle_image(image_data['path'])
        else:
            image_path = event.widget.image_path
            if self.current_design == "Ring":
                self.change_ring_image(image_path)
            elif self.current_design == "Earring":
                self.change_earring_image(image_path)
            elif self.current_design == "Bangle":
                self.change_bangle_image(image_path)


            

    def update_frame(self):
            import collections

            # Initialize a deque to store the last few positions
            self.ring_positions = collections.deque(maxlen=15)
            if not hasattr(self, 'left_ear_positions'):
                self.left_ear_positions = collections.deque(maxlen=20)
            if not hasattr(self, 'right_ear_positions'):
                self.right_ear_positions = collections.deque(maxlen=20)

            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)

            if ret:
                height, width, _ = frame.shape
                
                # Use full frame width and height
                center_width = width
                center_section = frame
                left_section = frame[:, :0]  # Empty section
                right_section = frame[:, width:]  # Empty section
                
                # Convert the full frame from OpenCV BGR format to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Since we're using the full frame, we don't need to reconstruct it
                frame = center_section


                if self.current_design == "Necklace":
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.face_detection.process(frame_rgb)
                    original_frame = frame.copy()

                    if not hasattr(self, 'pose_detection'):
                        self.mp_pose = mp.solutions.pose
                        self.pose_detection = self.mp_pose.Pose()
                        self.face_mesh = self.mp_face_mesh.FaceMesh()

                    try:
                        if results.detections:
                            for detection in results.detections:
                                bboxC = detection.location_data.relative_bounding_box
                                ih, iw, _ = frame.shape
                                
                                xminC, yminC = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                                widthC, heightC = int(bboxC.width * iw), int(bboxC.height * ih)
                                xmaxC, ymaxC = xminC + widthC, yminC + heightC

                                # Enhanced size factors and vertical offsets
                                if self.necklace_type == 'large2':
                                    size_factor = 1.2
                                    vertical_offset = -50
                                elif self.necklace_type == 'large':
                                    size_factor = 1.1
                                    vertical_offset = -45
                                elif self.necklace_type == 'choker':
                                    size_factor = 0.9
                                    vertical_offset = -130
                                else:  # regular
                                    size_factor = 0.9
                                    vertical_offset = -90

                                shoulder_ymin = ymaxC + (30 if self.necklace_type in ['large', 'large2'] else 15)
                                chest_ymax = min(ymaxC + (250 if self.necklace_type in ['large', 'large2'] else 200), ih)
                                
                                landmarks = self.pose_detection.process(frame_rgb)
                                if landmarks.pose_landmarks:
                                    left_shoulder = landmarks.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                                    right_shoulder = landmarks.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                                    
                                    shoulder_angle = np.arctan2(right_shoulder.y - left_shoulder.y,
                                                            abs(right_shoulder.x - left_shoulder.x))
                                    rotation_angle = np.degrees(shoulder_angle)
                                    
                                    # Increased base size multipliers
                                    necklace_width = int((xmaxC - xminC) * 1.2 * size_factor)
                                    necklace_height = int((chest_ymax - shoulder_ymin) * 1.2 * size_factor)
                                    resized_necklace = cv2.resize(self.necklace_image, (necklace_width, necklace_height))
                                    
                                    rotated_necklace = self.rotate_image(resized_necklace, rotation_angle)
                                    flipped_necklace = cv2.flip(rotated_necklace, 1)
                                    
                                    face_mesh_results = self.face_mesh.process(frame_rgb)
                                    if face_mesh_results.multi_face_landmarks:
                                        face_landmarks = face_mesh_results.multi_face_landmarks[0]
                                        nose_tip = face_landmarks.landmark[1]
                                        nose_x, nose_y = int(nose_tip.x * iw), int(nose_tip.y * ih)
                                        
                                        mid_shoulder_x = int((left_shoulder.x + right_shoulder.x) * iw / 2)
                                        mid_shoulder_y = int((left_shoulder.y + right_shoulder.y) * ih / 2)
                                        
                                        nose_angle = np.arctan2(nose_y - mid_shoulder_y, nose_x - mid_shoulder_x)
                                        
                                        # Adjusted positioning factors
                                        adjustment_factor = 0.25
                                        necklace_start_x = max(0, (mid_shoulder_x - flipped_necklace.shape[1] // 2) - 5)
                                        necklace_start_y = max(0, mid_shoulder_y + vertical_offset)
                                        
                                        necklace_start_x += int(np.cos(nose_angle) * adjustment_factor * flipped_necklace.shape[1])
                                        necklace_start_y += int(np.sin(nose_angle) * adjustment_factor * flipped_necklace.shape[0])
                                        
                                        necklace_end_x = min(frame.shape[1], necklace_start_x + flipped_necklace.shape[1])
                                        necklace_end_y = necklace_start_y + flipped_necklace.shape[0]
                                        
                                        region = frame[necklace_start_y:necklace_end_y, necklace_start_x:necklace_end_x]
                                        if region.shape[1] > 0 and region.shape[0] > 0:
                                            overlay_rgb = flipped_necklace[:, :, :3]
                                            mask = flipped_necklace[:, :, 3] / 255.0
                                            
                                            resized_mask = cv2.resize(mask, (region.shape[1], region.shape[0]))
                                            resized_overlay_rgb = cv2.resize(overlay_rgb, (region.shape[1], region.shape[0]))
                                            
                                            blended = (resized_overlay_rgb * resized_mask[:, :, np.newaxis] +
                                                    region * (1 - resized_mask[:, :, np.newaxis])).astype(np.uint8)
                                            
                                            frame[necklace_start_y:necklace_end_y, necklace_start_x:necklace_end_x] = blended
                                    
                    except Exception as e:
                        print(f"Error in necklace processing: {str(e)}")
                        frame = original_frame


                    
                elif self.current_design == "Ring":
                    # Initialize Kalman filters if not already initialized
                    if not hasattr(self, 'position_kalman'):
                        self.position_kalman = cv2.KalmanFilter(4, 2)
                        self.position_kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                        self.position_kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
                        self.position_kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
                        self.position_kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.05
                        self.position_kalman.errorCovPost = np.eye(4, dtype=np.float32)
                        self.position_kalman.statePost = np.zeros((4, 1), dtype=np.float32)

                        self.size_kalman = cv2.KalmanFilter(2, 1)
                        self.size_kalman.measurementMatrix = np.array([[1, 0]], np.float32)
                        self.size_kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
                        self.size_kalman.processNoiseCov = np.eye(2, dtype=np.float32) * 0.005
                        self.size_kalman.measurementNoiseCov = np.array([[0.05]], np.float32)
                        self.size_kalman.errorCovPost = np.eye(2, dtype=np.float32)
                        self.size_kalman.statePost = np.array([[50], [0]], np.float32)

                        self.center_x_smooth = None
                        self.center_y_smooth = None

                    # Perform hand detection on the center section
                    hand_results = self.hands.process(frame_rgb)

                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            # Get key landmarks for ring placement
                            mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
                            pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]

                            # Convert normalized coordinates to pixel values
                            mcp_x, mcp_y = int(mcp.x * frame.shape[1]), int(mcp.y * frame.shape[0])
                            pip_x, pip_y = int(pip.x * frame.shape[1]), int(pip.y * frame.shape[0])
                            
                            # Apply Kalman filter for smoothing position
                            measured_position = np.array([[np.float32((mcp_x + pip_x) / 2)], 
                                                        [np.float32((mcp_y + pip_y) / 2)]], np.float32)
                            self.position_kalman.correct(measured_position)
                            predicted_position = self.position_kalman.predict()

                            predicted_center_x = int(predicted_position[0])
                            predicted_center_y = int(predicted_position[1])

                            # Apply exponential smoothing
                            smoothing_alpha = 0.6
                            if self.center_x_smooth is None:
                                self.center_x_smooth = predicted_center_x
                                self.center_y_smooth = predicted_center_y
                            else:
                                self.center_x_smooth = int(smoothing_alpha * self.center_x_smooth + 
                                                        (1 - smoothing_alpha) * predicted_center_x)
                                self.center_y_smooth = int(smoothing_alpha * self.center_y_smooth + 
                                                        (1 - smoothing_alpha) * predicted_center_y)

                            # Calculate ring size using Kalman filter
                            distance = np.sqrt((pip_x - mcp_x) ** 2 + (pip_y - mcp_y) ** 2)
                            scaling_factor = 0.5
                            measured_size = np.array([[np.float32(distance * scaling_factor)]], np.float32)
                            self.size_kalman.correct(measured_size)
                            predicted_size = self.size_kalman.predict()
                            ring_size = max(20, min(100, int(predicted_size[0])))

                            # Calculate rotation angle
                            angle = -np.degrees(np.arctan2(pip_y - mcp_y, pip_x - mcp_x)) - 90

                            # Resize and rotate ring image
                            resized_ring = cv2.resize(self.ring_image, (ring_size, ring_size))
                            
                            # Rotate image with alpha channel
                            h, w = resized_ring.shape[:2]
                            center = (w // 2, h // 2)
                            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                            rotated_rgb = cv2.warpAffine(resized_ring[:, :, :3], rotation_matrix, (w, h))
                            rotated_alpha = cv2.warpAffine(resized_ring[:, :, 3], rotation_matrix, (w, h))
                            rotated_ring = cv2.merge([rotated_rgb, np.expand_dims(rotated_alpha, axis=2)])

                            # Calculate overlay position
                            x1 = self.center_x_smooth - ring_size // 2
                            y1 = self.center_y_smooth - ring_size // 2
                            x2 = x1 + ring_size
                            y2 = y1 + ring_size

                            # Check boundaries and apply overlay
                            if (x1 >= 0 and y1 >= 0 and 
                                x2 < center_section.shape[1] and 
                                y2 < center_section.shape[0]):
                                
                                roi_ring = center_section[y1:y2, x1:x2]
                                ring_alpha = rotated_ring[:, :, 3] / 255.0
                                mask_ring = np.stack([ring_alpha] * 3, axis=2)

                                if roi_ring.shape == mask_ring.shape:
                                    masked_ring = rotated_ring[:, :, :3] * mask_ring
                                    roi_mask_ring = 1 - mask_ring
                                    roi_combined_ring = roi_ring * roi_mask_ring
                                    combined_ring = cv2.add(masked_ring, roi_combined_ring)
                                    center_section[y1:y2, x1:x2] = combined_ring
                                    
                elif self.current_design == "Earring":
                    # Initialize Kalman filters if not already initialized
                    if not hasattr(self, 'kalman_left'):
                        self.kalman_left = KalmanFilter()
                        self.kalman_right = KalmanFilter()

                    # Perform face mesh detection on the center section
                    results = self.face_mesh.process(frame_rgb)

                    # Index numbers for the left and right ear landmarks
                    left_ear_index = 177
                    right_ear_index = 401
                    nose_index = 1
                    left_cheek = 234
                    right_cheek = 454

                    # Flags to check if both left and right earrings are detected
                    left_ear_detected = False
                    right_ear_detected = False
                    left_ear_x = left_ear_y = right_ear_x = right_ear_y = 0
                    left_ear_bbox_size = right_ear_bbox_size = 0

                    if results.multi_face_landmarks:
                        facial_landmarks = results.multi_face_landmarks[0]

                        # Get landmarks for nose, ears, and cheeks
                        nose_landmark = facial_landmarks.landmark[nose_index]
                        left_ear_landmark = facial_landmarks.landmark[left_ear_index]
                        right_ear_landmark = facial_landmarks.landmark[right_ear_index]
                        left_cheek_landmark = facial_landmarks.landmark[left_cheek]
                        right_cheek_landmark = facial_landmarks.landmark[right_cheek]

                        # Calculate face width for dynamic sizing
                        face_width = abs(right_cheek_landmark.x - left_cheek_landmark.x)
                        distance_scale = face_width / 0.3
                        base_size = int(20 * distance_scale)

                        # Calculate head turn angle
                        nose_x = nose_landmark.x
                        left_ear_x = left_ear_landmark.x
                        right_ear_x = right_ear_landmark.x
                        head_turn_angle = np.degrees(np.arctan2(nose_x - (left_ear_x + right_ear_x) / 2, 0.5))

                        # Define turn_offset here so it's available for both ear processing blocks
                        turn_offset = int(200 * abs(head_turn_angle) / 90)

                        # Get original aspect ratio of earring image
                        orig_height, orig_width = self.earring_image.shape[:2]
                        orig_aspect_ratio = orig_width / orig_height

                        # Process left ear
                        if head_turn_angle >= -5:
                            left_ear_x = int(left_ear_landmark.x * frame.shape[1])
                            left_ear_y = int(left_ear_landmark.y * frame.shape[0]) 
                            
                            # Apply Kalman filter
                            left_ear_x, left_ear_y = self.kalman_left.update(left_ear_x, left_ear_y)
                            
                            # Calculate bounding box with dynamic sizing
                            base_size = int(15 * distance_scale)
                            left_ear_bbox_size = base_size
                            
                            # Calculate outward movement based on head turn
                            turn_offset = int(200 * abs(head_turn_angle) / 90)
                            left_ear_offset = (- turn_offset, 5 if head_turn_angle > 10 else 5)
                            
                            # Apply position updates
                            left_ear_x += left_ear_offset[0]
                            left_ear_y += left_ear_offset[1]

                            # Update positions buffer and calculate averages
                            self.left_ear_positions.append((left_ear_x, left_ear_y, left_ear_bbox_size))
                            avg_left_ear_x = int(sum(pos[0] for pos in self.left_ear_positions) / len(self.left_ear_positions))
                            avg_left_ear_y = int(sum(pos[1] for pos in self.left_ear_positions) / len(self.left_ear_positions))
                            avg_left_ear_bbox_size = int(sum(pos[2] for pos in self.left_ear_positions) / len(self.left_ear_positions))

                            # Calculate final positions
                            left_ear_top_left = (avg_left_ear_x - avg_left_ear_bbox_size, avg_left_ear_y - avg_left_ear_bbox_size)
                            left_ear_bottom_right = (avg_left_ear_x + avg_left_ear_bbox_size, avg_left_ear_y + avg_left_ear_bbox_size)

                            # Apply earring overlay with aspect ratio preservation
                            if (left_ear_top_left[0] >= 0 and left_ear_top_left[1] >= 0 and 
                                left_ear_bottom_right[0] <= frame.shape[1] and 
                                left_ear_bottom_right[1] <= frame.shape[0]):
                                
                                ear_width = left_ear_bottom_right[0] - left_ear_top_left[0]
                                ear_height = left_ear_bottom_right[1] - left_ear_top_left[1]
                                
                                target_height = ear_height
                                target_width = int(target_height * orig_aspect_ratio)
                                
                                if target_width > ear_width:
                                    target_width = ear_width
                                    target_height = int(target_width / orig_aspect_ratio)
                                
                                resized_earring = cv2.resize(self.earring_image, (target_width, target_height))
                                
                                # Create centered overlay
                                final_earring = np.zeros((ear_height, ear_width, 4), dtype=np.uint8)
                                x_offset = (ear_width - target_width) // 2
                                y_offset = (ear_height - target_height) // 2
                                final_earring[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = resized_earring

                                # Apply transparency
                                alpha_channel = final_earring[:, :, 3]
                                mask = alpha_channel[:, :, np.newaxis] / 255.0
                                overlay = final_earring[:, :, :3] * mask
                                mask_inv = 1 - mask
                                region_left = frame[left_ear_top_left[1]:left_ear_bottom_right[1],
                                                left_ear_top_left[0]:left_ear_bottom_right[0]]
                                region_left_inv = region_left * mask_inv
                                region_left_combined = cv2.add(overlay.astype(np.uint8), region_left_inv.astype(np.uint8))
                                frame[left_ear_top_left[1]:left_ear_bottom_right[1],
                                    left_ear_top_left[0]:left_ear_bottom_right[0]] = region_left_combined
                                left_ear_detected = True

                        # Process right ear
                        if head_turn_angle <= 5:
                            right_ear_x = int(right_ear_landmark.x * frame.shape[1])
                            right_ear_y = int(right_ear_landmark.y * frame.shape[0])
                            
                            # Apply Kalman filter for right ear
                            right_ear_x, right_ear_y = self.kalman_right.update(right_ear_x, right_ear_y)
                            
                            # Calculate bounding box with dynamic sizing
                            right_ear_bbox_size = base_size
                            
                            # Calculate outward movement based on head turn
                            right_ear_offset = (turn_offset, 5 if head_turn_angle > 10 else 5)
                            
                            # Apply position updates
                            right_ear_x += right_ear_offset[0]
                            right_ear_y += right_ear_offset[1]

                            # Update positions buffer and calculate averages
                            self.right_ear_positions.append((right_ear_x, right_ear_y, right_ear_bbox_size))
                            avg_right_ear_x = int(sum(pos[0] for pos in self.right_ear_positions) / len(self.right_ear_positions))
                            avg_right_ear_y = int(sum(pos[1] for pos in self.right_ear_positions) / len(self.right_ear_positions))
                            avg_right_ear_bbox_size = int(sum(pos[2] for pos in self.right_ear_positions) / len(self.right_ear_positions))

                            # Calculate final positions
                            right_ear_top_left = (avg_right_ear_x - avg_right_ear_bbox_size, avg_right_ear_y - avg_right_ear_bbox_size)
                            right_ear_bottom_right = (avg_right_ear_x + avg_right_ear_bbox_size, avg_right_ear_y + avg_right_ear_bbox_size)

                            # Apply earring overlay with aspect ratio preservation
                            if (right_ear_top_left[0] >= 0 and right_ear_top_left[1] >= 0 and 
                                right_ear_bottom_right[0] <= frame.shape[1] and 
                                right_ear_bottom_right[1] <= frame.shape[0]):
                                
                                ear_width = right_ear_bottom_right[0] - right_ear_top_left[0]
                                ear_height = right_ear_bottom_right[1] - right_ear_top_left[1]
                                
                                target_height = ear_height
                                target_width = int(target_height * orig_aspect_ratio)
                                
                                if target_width > ear_width:
                                    target_width = ear_width
                                    target_height = int(target_width / orig_aspect_ratio)
                                
                                resized_earring = cv2.resize(self.earring_image, (target_width, target_height))
                                
                                # Create centered overlay
                                final_earring = np.zeros((ear_height, ear_width, 4), dtype=np.uint8)
                                x_offset = (ear_width - target_width) // 2
                                y_offset = (ear_height - target_height) // 2
                                final_earring[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = resized_earring

                                # Apply transparency
                                alpha_channel = final_earring[:, :, 3]
                                mask = alpha_channel[:, :, np.newaxis] / 255.0
                                overlay = final_earring[:, :, :3] * mask
                                mask_inv = 1 - mask
                                region_right = frame[right_ear_top_left[1]:right_ear_bottom_right[1],
                                                right_ear_top_left[0]:right_ear_bottom_right[0]]
                                region_right_inv = region_right * mask_inv
                                region_right_combined = cv2.add(overlay.astype(np.uint8), region_right_inv.astype(np.uint8))
                                frame[right_ear_top_left[1]:right_ear_bottom_right[1],
                                    right_ear_top_left[0]:right_ear_bottom_right[0]] = region_right_combined
                                right_ear_detected = True

                        # Track average coordinates and dimensions
                        alpha = 0.3
                        if not hasattr(self, 'avg_center_x'):
                            self.avg_center_x = (left_ear_x + right_ear_x) // 2
                            self.avg_center_y = (left_ear_y + right_ear_y) // 2
                            self.avg_width = right_ear_x - left_ear_x
                            self.avg_height = (left_ear_bbox_size + right_ear_bbox_size) // 2
                        else:
                            self.avg_center_x = int(self.avg_center_x * (1 - alpha) + (left_ear_x + right_ear_x) // 2 * alpha)
                            self.avg_center_y = int(self.avg_center_y * (1 - alpha) + (left_ear_y + right_ear_y) // 2 * alpha)
                            self.avg_width = int(self.avg_width * (1 - alpha) + (right_ear_x - left_ear_x) * alpha)
                            self.avg_height = int(self.avg_height * (1 - alpha) + ((left_ear_bbox_size + right_ear_bbox_size) // 2) * alpha)



                elif self.current_design == "Bangle":
                    # Perform hand detection on the center section
                    # Perform hand detection on the center section
                    hand_results = self.hands.process(frame_rgb)

                    # Check if any hands were detected in the center section
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            # Get the coordinates of the wrist (point 0)
                            wrist = hand_landmarks.landmark[0]
                            wrist_x = int(wrist.x * center_section.shape[1]) - 5
                            wrist_y = int(wrist.y * center_section.shape[0]) + 30  # Adjust the y-coordinate by adding the offset

                            # Add the current wrist position to the buffer
                            self.hand_positions.append((wrist_x, wrist_y))

                            # Calculate the average wrist position
                            avg_wrist_x = int(np.mean([pos[0] for pos in self.hand_positions]))
                            avg_wrist_y = int(np.mean([pos[1] for pos in self.hand_positions]))

                            # Define the bounding box parameters
                            box_width = center_section.shape[1]  # Use full width of center section
                            box_height = center_section.shape[0]  # Use full height of center section
                            half_width = box_width // 2
                            half_height = box_height // 2
                            top_left = (avg_wrist_x - half_width, avg_wrist_y - half_height)
                            bottom_right = (avg_wrist_x + half_width, avg_wrist_y + half_height)

                            # Check if the bounding box is within the frame
                            if top_left[0] >= 0 and top_left[1] >= 0 and bottom_right[0] < center_section.shape[1] and bottom_right[1] < center_section.shape[0]:
                                # Resize the bangle image to fit the bounding box size
                                resized_bangle = cv2.resize(self.bangle_image, (box_width, box_height))

                                # Define the region of interest for placing the bangle image
                                roi = center_section[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                                # Create a mask from the alpha channel of the bangle image
                                bangle_alpha = resized_bangle[:, :, 3] / 255.0
                                mask = np.stack([bangle_alpha] * 3, axis=2)

                                # Apply the mask to the bangle image
                                masked_bangle = resized_bangle[:, :, :3] * mask

                                # Create a mask for the region of interest
                                roi_mask = 1 - mask

                                # Apply the inverse mask to the region of interest
                                roi_combined = roi * roi_mask

                                # Combine the masked bangle image and the region of interest
                                combined = cv2.add(masked_bangle, roi_combined)

                                # Replace the region of interest with the combined image
                                center_section[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = combined



                    
                    else:
                            # Display text message when no face is detected
                            if center_section is not None and center_section.shape[0] > 0 and center_section.shape[1] > 0:
                                # Define the text and its properties
                                text = "Not detectable"
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 1
                                color = (0, 0, 255)  # Red color in BGR
                                thickness = 2

                                # Get the text size
                                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                                # Calculate the text position
                                text_x = (center_section.shape[1] - text_size[0])//2;
                                text_y = (center_section.shape[0] + text_size[1])//2;

                                # Put the text on the image
                                cv2.putText(center_section, text, (text_x, text_y), font, font_scale, color, thickness)
                            else:
                                print("center_section is None or empty. Cannot display text.")
                                        

                # No need to merge sections since we're using the full frame
                frame = center_section


                frame = cv2.resize(frame, (650, 650))
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image_tk = ImageTk.PhotoImage(image=image)
                self.webcam_canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
                self.webcam_canvas.image_tk = image_tk

                self.root.after(10, self.update_frame)
            else:
                self.cap.release()

# Create and run the application
root = tk.Tk()
app = VirtualTryOnApp(root)
root.mainloop()
