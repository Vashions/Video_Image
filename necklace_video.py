from necklace_image import advanced_necklace_extraction
from flask import request, jsonify
from collections import deque
import mediapipe as mp
import numpy as np
import base64
import cv2
import os
mp_pose = mp.solutions.pose

import logging
# Add these as global variables
COORDS_BUFFER_SIZE = 5
position_buffer = deque(maxlen=COORDS_BUFFER_SIZE)
angle_buffer = deque(maxlen=COORDS_BUFFER_SIZE)
# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pose_detection = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)




def calculate_jewelry_coordinates(frame, landmarks):
    height, width = frame.shape[:2]
    
    if landmarks and landmarks.pose_landmarks:
        left_shoulder = landmarks.pose_landmarks.landmark[11]
        right_shoulder = landmarks.pose_landmarks.landmark[12]
        x = (left_shoulder.x + right_shoulder.x) / 2
        y = (left_shoulder.y + right_shoulder.y) / 2 - 0.05  # Slightly above shoulders
        
        # Normalize coordinates
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        
        return {'x': x, 'y': y}
    
    return None



import cv2
import numpy as np
import logging

# Initialize logger
logger = logging.getLogger(__name__)

def process_necklace_design(frame, necklace_image, interpolated_landmarks=None, necklace_type='regular', position_buffer_size=5, movement_threshold=10):

    if frame is None or necklace_image is None:
        logger.error("Invalid frame or necklace image")
        return frame, None

    original_frame = frame.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    if results.detections is None:
        logger.warning("No face detected in the frame.")
        return frame, None

    # Necklace processing variables
    necklace_extracted = advanced_necklace_extraction(necklace_image)
    # Flip the necklace horizontally
    necklace_extracted = cv2.flip(necklace_extracted, 1)
    
    # Size factors based on necklace type
    size_factors = {
        'large2': (1.5, -30),
        'large': (1.3, -30),
        'choker': (0.9, -50),
        'regular': (1.0, -40),
    }
    size_factor, vertical_offset = size_factors.get(necklace_type, (1.0, -60))

    # Initialize position buffer and last stable position if not exists
    if not hasattr(process_necklace_design, "necklace_position_buffer"):
        process_necklace_design.necklace_position_buffer = []
    if not hasattr(process_necklace_design, "last_stable_position"):
        process_necklace_design.last_stable_position = None

    try:
        for detection in results.detections:
            # Get face bounding box
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            xminC, yminC = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            widthC, heightC = int(bboxC.width * iw), int(bboxC.height * ih)
            xmaxC, ymaxC = xminC + widthC, yminC + heightC

            # Calculate necklace region
            shoulder_ymin = ymaxC + (30 if necklace_type in ['large', 'large2'] else 15)
            chest_ymax = min(ymaxC + (250 if necklace_type in ['large', 'large2'] else 200), ih)
            xminC -= 40 if necklace_type in ['large', 'large2'] else 30
            xmaxC += 40 if necklace_type in ['large', 'large2'] else 30

            # Size calculations
            necklace_width = int((xmaxC - xminC) * 0.8 * size_factor)
            necklace_height = int((chest_ymax - shoulder_ymin) * 0.8 * size_factor)
            
            # Only resize if necessary (to fit within the calculated region)
            resized_necklace = cv2.resize(necklace_extracted, (necklace_width, necklace_height))

            # Position stabilization
            mid_shoulder_x = (xminC + xmaxC) // 2
            mid_shoulder_y = shoulder_ymin
            current_position = (mid_shoulder_x, mid_shoulder_y, necklace_width, necklace_height)

            # Update position buffer
            buffer = process_necklace_design.necklace_position_buffer
            buffer.append(current_position)
            if len(buffer) > position_buffer_size:
                buffer.pop(0)

            # Calculate average position
            avg_position = np.mean(buffer, axis=0).astype(int)
            avg_x, avg_y, avg_width, avg_height = avg_position
            avg_y += vertical_offset

            # Check for significant movement
            if process_necklace_design.last_stable_position:
                last_x, last_y, _, _ = process_necklace_design.last_stable_position
                movement = np.sqrt((avg_x - last_x)**2 + (avg_y - last_y)**2)

                if movement < movement_threshold:
                    avg_x, avg_y, avg_width, avg_height = process_necklace_design.last_stable_position
                else:
                    process_necklace_design.last_stable_position = (avg_x, avg_y, avg_width, avg_height)
            else:
                process_necklace_design.last_stable_position = (avg_x, avg_y, avg_width, avg_height)

            # Calculate final placement coordinates
            necklace_end_x = min(frame.shape[1], avg_x + avg_width // 2)
            necklace_start_x = max(0, avg_x - avg_width // 2)
            necklace_end_y = min(frame.shape[0], avg_y + avg_height)
            necklace_start_y = max(0, avg_y)

            # Apply necklace overlay
            region = frame[necklace_start_y:necklace_end_y, necklace_start_x:necklace_end_x]
            if region.shape[0] > 0 and region.shape[1] > 0:
                overlay_rgb = resized_necklace[:, :, :3]  # Extract RGB channels
                mask = resized_necklace[:, :, 3] / 255.0  # Alpha channel for blending

                resized_mask = cv2.resize(mask, (region.shape[1], region.shape[0]))
                resized_overlay_rgb = cv2.resize(overlay_rgb, (region.shape[1], region.shape[0]))

                # Blend the necklace onto the frame using the alpha mask for transparency
                blended = (resized_overlay_rgb * resized_mask[:, :, np.newaxis] +
                          region * (1 - resized_mask[:, :, np.newaxis])).astype(np.uint8)
                frame[necklace_start_y:necklace_end_y, necklace_start_x:necklace_end_x] = blended

    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        frame = original_frame

    return frame, process_necklace_design.necklace_position_buffer


def rotate_image(image, angle):
    # Add padding around the image before rotating to avoid clipping
    height, width = image.shape[:2]
    diagonal = int(np.sqrt(height**2 + width**2))
    
    # Create a padded square canvas to prevent any clipping during rotation
    padded_image = np.zeros((diagonal, diagonal, image.shape[2]), dtype=np.uint8)
    x_offset = (diagonal - width) // 2
    y_offset = (diagonal - height) // 2
    
    # Place the original image in the center of the padded canvas
    padded_image[y_offset:y_offset + height, x_offset:x_offset + width] = image
    
    # Perform the rotation
    center = (diagonal // 2, diagonal // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)  # Negative angle for opposite rotation
    rotated_image = cv2.warpAffine(padded_image, rotation_matrix, (diagonal, diagonal), flags=cv2.INTER_LINEAR)
    
    return rotated_image


