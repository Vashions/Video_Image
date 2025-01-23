from flask import Flask, flash, request, redirect, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from earring_image import earring_process_images
from ultralytics import YOLO
from io import BytesIO
import mediapipe as mp
import numpy as np
import base64
import cv2
import io
import os

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image Inputs for Necklace
NecklaceFolderPath = "static/Image/Necklace"
listnecklace = os.listdir(NecklaceFolderPath)
necklace_image = cv2.imread(os.path.join(NecklaceFolderPath, listnecklace[0]), cv2.IMREAD_UNCHANGED)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)



def detect_shoulders(pose_landmarks, img_shape):
    """
    Detect the shoulder landmarks and return their coordinates.
    """
    ih, iw, _ = img_shape

    # MediaPipe Pose Landmarks for shoulders
    left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]

    # Convert normalized coordinates to pixel values
    left_shoulder = (int(left_shoulder.x * iw), int(left_shoulder.y * ih))
    right_shoulder = (int(right_shoulder.x * iw), int(right_shoulder.y * ih))

    return left_shoulder, right_shoulder

def calculate_rotation_angle(left_shoulder, right_shoulder):
    """
    Calculate the angle to rotate the necklace based on the shoulder alignment.
    Adjust to ensure the necklace is upright.
    """
    # Calculate the height difference between shoulders
    height_difference = right_shoulder[1] - left_shoulder[1]

    angle = height_difference * 0.1  

    return angle

def rotate_image(image, angle):
    """
    Rotate an image by a given angle while preserving size.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    return rotated_image


def get_scale_factors(neck_width): 
    ranges = [
        (0, 50), (50, 80), (80, 100), (100, 130), (130, 160),
        (160, 200), (200, 250), (250, 300), (300, 350), (350, 400),
        (400, 500), (500, 600), (600, 700), (700, 800), (800, 900),
        (900, 1000), (1000, 1100), (1100, 1200), (1200, 1300), (1300, 1400),
        (1400, 1500)
    ]
    
    factors = [
        (0.30, 0.50), (0.25, 0.40), (0.3, 0.50), (0.40, 0.55), (0.40, 0.60),
        (0.50, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 0.85), (0.75, 0.90),
        (0.80, 0.95), (0.85, 1.00), (0.90, 1.05), (1.10, 1.20), (1.20, 1.20),
        (1.35, 1.35), (1.32, 1.35), (1.40, 1.40), (1.50, 1.45), (1.52, 1.54),
        (1.65, 1.65)
    ]

    for (start, end), (scale, reduction) in zip(ranges, factors):
        if start <= neck_width < end:
            return scale, reduction

    return 1.6, 1.6  # Default factors for out-of-range sizes


def detect_neck_center(pose_landmarks, img_shape):
    """
    Detect the neck center based on the shoulder positions.
    """
    ih, iw, _ = img_shape

    left_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    
    neck_center_x = int((left_shoulder.x + right_shoulder.x) / 2 * iw)
    neck_center_y = int((left_shoulder.y + right_shoulder.y) / 2 * ih)

    return (neck_center_x, neck_center_y)



mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def visualize_face_mesh(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = img.shape
            
            # Loop through all the landmarks and draw them on the image
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * iw)
                y = int(landmark.y * ih)
                cv2.circle(img, (x, y), 1, (255, 0, 0), -1)  # Draw landmarks in red
            
            # Draw specific landmarks for visualization (jawline and neck)
            for i in range(0, 17):  # Jawline points
                landmark = face_landmarks.landmark[i]
                x = int(landmark.x * iw)
                y = int(landmark.y * ih)
                cv2.circle(img, (x, y), 3, (0, 255, 0), -1)  # Jawline in green
            
            # Neck points (indices may need to be adjusted based on your needs)
            for i in range(152, 156):  # Adjust based on your needs
                landmark = face_landmarks.landmark[i]
                x = int(landmark.x * iw)
                y = int(landmark.y * ih)
                cv2.circle(img, (x, y), 3, (255, 0, 255), -1)  # Neck points in magenta

    return img





def detect_and_overlay_accessory(image_bytes_io, jewelry_image=None, jewelry_type='necklace', jewelry_filename=''):
    image_array = np.frombuffer(image_bytes_io.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if jewelry_image is not None:
        jewelry_image_resized = cv2.resize(jewelry_image, (800, 800), interpolation=cv2.INTER_LANCZOS4)
    else:
        return img

    # Necklace logic
    if jewelry_type == 'necklace':
        necklace_type = 'regular'
        if '_large2' in jewelry_filename.lower():
            necklace_type = 'large2'
        elif '_large' in jewelry_filename.lower():
            necklace_type = 'large'
        elif '_choker' in jewelry_filename.lower():
            necklace_type = 'choker'

        print(f"Processing a {necklace_type} necklace design")

        # Adjusted size factors and vertical offsets
        size_factor, vertical_offset_factor = {
            'large2': (0.9, 0.3),
            'large': (0.8, 0.4),
            'choker': (0.55, 0.65),
            'regular': (0.60, 0.50)
        }.get(necklace_type, (0.85, 0.4))

        # MediaPipe Pose initialization
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks
            left_shoulder, right_shoulder = detect_shoulders(pose_landmarks, img.shape)
            neck_center = detect_neck_center(pose_landmarks, img.shape)

            # # Visualize shoulder and neck center landmarks
            # cv2.circle(img, left_shoulder, 5, (0, 255, 0), -1)    # Green dot for left shoulder
            # cv2.circle(img, right_shoulder, 5, (255, 0, 0), -1)   # Blue dot for right shoulder
            # cv2.circle(img, neck_center, 5, (0, 0, 255), -1)      # Red dot for neck center

            rotation_angle = calculate_rotation_angle(left_shoulder, right_shoulder)

            shoulder_distance = abs(right_shoulder[0] - left_shoulder[0])
            new_width = int(shoulder_distance * size_factor)
            new_height = int(jewelry_image_resized.shape[0] * (new_width / jewelry_image_resized.shape[1]))
            resized_necklace = cv2.resize(jewelry_image_resized, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

            # Calculate the rotation matrix and new bounding box dimensions
            center = (new_width // 2, new_height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
            cos = abs(rotation_matrix[0, 0])
            sin = abs(rotation_matrix[0, 1])

            # Compute new bounding dimensions to fit the rotated necklace
            bound_w = int((new_height * sin) + (new_width * cos))
            bound_h = int((new_height * cos) + (new_width * sin))

            # Adjust the rotation matrix to the new bounding size
            rotation_matrix[0, 2] += bound_w / 2 - center[0]
            rotation_matrix[1, 2] += bound_h / 2 - center[1]

            # Rotate the necklace image with adjusted bounding box
            rotated_necklace = cv2.warpAffine(resized_necklace, rotation_matrix, (bound_w, bound_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

            # Adjust necklace position to be slightly lower
            start_x = int(neck_center[0] - bound_w / 2)
            start_y = int(neck_center[1] - bound_h * vertical_offset_factor) + 20
            end_x = start_x + bound_w
            end_y = start_y + bound_h

            # Ensure bounds
            start_x, start_y = max(0, start_x), max(0, start_y)
            end_x, end_y = min(img.shape[1], end_x), min(img.shape[0], end_y)

            # Alpha channel extraction for blending
            overlay_resized = rotated_necklace[:end_y - start_y, :end_x - start_x]
            alpha_channel = overlay_resized[:, :, 3] / 255.0
            mask = np.dstack([alpha_channel] * 3)
            region = img[start_y:end_y, start_x:end_x]

            # Blending necklace onto the image
            background = region * (1 - mask)
            img[start_y:end_y, start_x:end_x] = overlay_resized[:, :, :3] * mask + background

    # Earring logic
    elif jewelry_type == 'earring':
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(img_rgb)

        jewelry_image = remove_background(jewelry_image)

        size_factor = 0.25  # Smaller size factor for earrings
        if result.multi_face_landmarks:
            ih, iw, _ = img.shape
            for face_landmarks in result.multi_face_landmarks:
                for point_index, is_left_ear in [(177, True), (401, False)]:
                    point = tuple(map(int, (face_landmarks.landmark[point_index].x * iw, face_landmarks.landmark[point_index].y * ih)))

                    face_width = int(abs(face_landmarks.landmark[454].x - face_landmarks.landmark[234].x) * iw)
                    earring_height = int(face_width * size_factor)
                    aspect_ratio = jewelry_image.shape[1] / jewelry_image.shape[0]
                    earring_width = int(earring_height * aspect_ratio)

                    x_offset = -earring_width if is_left_ear else 0
                    y_offset = -earring_height // 2 - 10
                    center_x = point[0] + x_offset
                    center_y = point[1] + y_offset

                    center_x = max(0, min(center_x, iw - earring_width))
                    center_y = max(0, min(center_y, ih - earring_height))

                    resized_earring = cv2.resize(jewelry_image, (earring_width, earring_height))

                    if resized_earring.shape[2] == 4:
                        alpha_channel = resized_earring[:, :, 3]
                        mask = alpha_channel[:, :, np.newaxis] / 255.0
                    else:
                        mask = np.ones_like(resized_earring[:, :, :3])

                    mask = np.dstack([mask] * 3)
                    roi = img[center_y:center_y+earring_height, center_x:center_x+earring_width]

                    if roi.shape[:2] != resized_earring[:, :, :3].shape[:2]:
                        continue

                    blended = (mask * resized_earring[:, :, :3] + (1 - mask) * roi).astype(np.uint8)
                    img[center_y:center_y+earring_height, center_x:center_x+earring_width] = blended

    elif jewelry_type == 'ring':
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            ih, iw, _ = img.shape

            # Process each hand detected
            for hand_landmarks in results.multi_hand_landmarks:
                # Get key landmarks for ring placement (MCP, PIP, DIP)
                mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
                dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]

                # Calculate the distance between MCP and DIP to estimate finger length
                mcp_x, mcp_y = mcp.x * iw, mcp.y * ih
                pip_x, pip_y = pip.x * iw, pip.y * ih
                dip_x, dip_y = dip.x * iw, dip.y * ih

                # Calculate finger length (MCP to DIP)
                finger_length = np.sqrt((mcp_x - dip_x) ** 2 + (mcp_y - dip_y) ** 2)

                # Increase the scaling factor to make the ring larger
                scaling_factor = 0.3  # Make the ring larger, you can adjust this value

                # Use finger length to adjust ring size (finger size-based resizing)
                ring_height = int(finger_length * scaling_factor)
                aspect_ratio = jewelry_image.shape[1] / jewelry_image.shape[0]
                ring_width = int(ring_height * aspect_ratio)

                # Calculate position for ring based on MCP (base of the finger) and PIP (next joint)
                mcp_x = int(mcp.x * iw)
                mcp_y = int(mcp.y * ih)
                pip_x = int(pip.x * iw)
                pip_y = int(pip.y * ih)

                # Compute the center between MCP and PIP for the ring's position
                ring_x = (mcp_x + pip_x) // 2
                ring_y = (mcp_y + pip_y) // 2

                # Position the ring at the center of the MCP and PIP
                vertical_offset = -20  # Adjust this value to position the ring slightly below the finger
                start_x = ring_x - ring_width // 2
                start_y = max(0, ring_y + vertical_offset - ring_height // 2)

                # Ensure bounds
                start_x = max(0, min(start_x, iw - ring_width))
                start_y = max(0, min(start_y, ih - ring_height))

                # Resize the ring image
                resized_ring = cv2.resize(jewelry_image, (ring_width, ring_height), interpolation=cv2.INTER_AREA)

                # Overlay ring image
                if resized_ring.shape[2] == 4:  # If the ring has an alpha channel
                    alpha_channel = resized_ring[:, :, 3] / 255.0
                    mask = np.dstack([alpha_channel] * 3)
                    region = img[start_y:start_y + ring_height, start_x:start_x + ring_width]

                    # Ensure the region matches the resized ring dimensions
                    if region.shape[:2] != resized_ring[:, :, :3].shape[:2]:
                        continue

                    blended = (mask * resized_ring[:, :, :3] + (1 - mask) * region).astype(np.uint8)
                    img[start_y:start_y + ring_height, start_x:start_x + ring_width] = blended
                else:
                    img[start_y:start_y + ring_height, start_x:start_x + ring_width] = resized_ring

    return img





def advanced_necklace_extraction(necklace_image):
    if necklace_image.shape[2] == 3:
        necklace_image = cv2.cvtColor(necklace_image, cv2.COLOR_BGR2BGRA)

    if np.any(necklace_image[:,:,3] < 255):
        return necklace_image

    hsv = cv2.cvtColor(necklace_image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 225])
    upper = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.bitwise_not(mask)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    result = necklace_image.copy()
    result[:,:,3] = mask

    return result







def remove_background(image):
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    mask = np.all(image[:, :, :3] > [240, 240, 240], axis=2)
    image[mask] = [0, 0, 0, 0]
    
    return image


def necklace_process_images(customer_image, image2, jewelry_filename=''):
    # Decode images from buffer
    customer_image1 = cv2.imdecode(np.frombuffer(customer_image, np.uint8), cv2.IMREAD_COLOR)
    necklace_image1 = cv2.imdecode(np.frombuffer(image2, np.uint8), cv2.IMREAD_UNCHANGED)

    # Ensure necklace image is 800x800
    necklace_image_resized = cv2.resize(necklace_image1, (800, 800), interpolation=cv2.INTER_LANCZOS4) \
        if necklace_image1.shape[:2] != (800, 800) else necklace_image1

    # Extract the necklace from background
    necklace_extracted = advanced_necklace_extraction(necklace_image_resized)

    # Categorize necklace type based on filename
    if '_large2' in jewelry_filename.lower():
        necklace_type = 'large2'
    elif '_large' in jewelry_filename.lower():
        necklace_type = 'large'
    elif '_choker' in jewelry_filename.lower():
        necklace_type = 'choker'
    else:
        necklace_type = 'regular'

    print(f"Detected necklace type: {necklace_type}")  # Debugging to confirm category

    # Set size and vertical offset factors based on type
    size_factor, vertical_offset = {
        'large2': (0.9, 0.3),
        'large': (0.8, 0.4),
        'choker': (0.55, 0.65),   # Adjusted choker size for more effectiveness
        'regular': (0.6, 0.5)
    }[necklace_type]

    print(f"Size factor: {size_factor}, Vertical offset: {vertical_offset}")  # Debugging

    # Initialize MediaPipe Pose for landmark detection
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        img_rgb = cv2.cvtColor(customer_image1, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

    if results.pose_landmarks:
        # Get shoulder and neck center points
        left_shoulder, right_shoulder = detect_shoulders(results.pose_landmarks, customer_image1.shape)
        neck_center = detect_neck_center(results.pose_landmarks, customer_image1.shape)

        # Calculate rotation and resizing
        rotation_angle = calculate_rotation_angle(left_shoulder, right_shoulder)
        shoulder_distance = abs(right_shoulder[0] - left_shoulder[0])

        print(f"Shoulder distance: {shoulder_distance}")  # Debugging

        # Apply size factor for resizing
        new_width = int(shoulder_distance * size_factor)
        new_height = int(necklace_extracted.shape[0] * (new_width / necklace_extracted.shape[1]))

        print(f"Resized necklace dimensions: {new_width}x{new_height}")  # Debugging

        # Resize and rotate the necklace
        resized_necklace = cv2.resize(necklace_extracted, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        center = (new_width // 2, new_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
        rotated_necklace = cv2.warpAffine(resized_necklace, rotation_matrix, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        # Position the necklace based on neck center and vertical offset
        start_x = int(neck_center[0] - new_width / 2)
        start_y = int(neck_center[1] - new_height * vertical_offset)

        # Adjust bounds to fit within the image
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(customer_image1.shape[1], start_x + new_width)
        end_y = min(customer_image1.shape[0], start_y + new_height)

        # Extract overlay region and alpha channel
        overlay_resized = rotated_necklace[:end_y - start_y, :end_x - start_x]
        alpha_channel = overlay_resized[:, :, 3] / 255.0
        mask = np.dstack([alpha_channel] * 3)
        region = customer_image1[start_y:end_y, start_x:end_x]

        # Blend necklace onto the customer image
        background = region * (1 - mask)
        customer_image1[start_y:end_y, start_x:end_x] = overlay_resized[:, :, :3] * mask + background

    return customer_image1