import cv2
import mediapipe as mp
import numpy as np
from necklace_image import advanced_necklace_extraction
from mediapipe.python.solutions.face_mesh import FaceMesh

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)



import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh

# Kalman Filter class for smoothing coordinates
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

# Function to process earring design
def process_earring_design(frame, earring_image, earring_type='medium', base_size_factor=2.0,
                           vertical_offset=-10, horizontal_offset=0, y_adjustment=20):
    # Load the earring image
    if isinstance(earring_image, str):
        earring_image = cv2.imread(earring_image, cv2.IMREAD_UNCHANGED)
    if earring_image is None:
        print(f"Failed to load earring image")
        return frame, None

    earring_image = advanced_necklace_extraction(earring_image)  # Extract transparency if needed

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # Facial landmarks for ear positions
    LEFT_EAR_INDEX = 177
    RIGHT_EAR_INDEX = 401
    NOSE_INDEX = 1
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454

    left_ear_detected = False
    right_ear_detected = False
    jewelry_coords = None
    left_earring_coords = None
    right_earring_coords = None
    frame_height, frame_width = frame.shape[:2]

    if results.multi_face_landmarks:
        facial_landmarks = results.multi_face_landmarks[0]  # Assuming one face

        # Get landmarks for nose, ears, and cheeks
        nose_landmark = facial_landmarks.landmark[NOSE_INDEX]
        left_ear_landmark = facial_landmarks.landmark[LEFT_EAR_INDEX]
        right_ear_landmark = facial_landmarks.landmark[RIGHT_EAR_INDEX]
        left_cheek_landmark = facial_landmarks.landmark[LEFT_CHEEK]
        right_cheek_landmark = facial_landmarks.landmark[RIGHT_CHEEK]

        # Calculate face width and use it for dynamic sizing
        face_width = abs(right_cheek_landmark.x - left_cheek_landmark.x)
        distance_scale = face_width / 0.3  # Base scaling factor

        # Calculate head turn angle
        nose_x = nose_landmark.x
        left_ear_x = left_ear_landmark.x
        right_ear_x = right_ear_landmark.x
        head_turn_angle = np.degrees(np.arctan2(nose_x - (left_ear_x + right_ear_x) / 2, 0.5))

        print(f"Head turn angle: {head_turn_angle:.2f} degrees, Face width: {face_width:.3f}, Scale: {distance_scale:.3f}")

        # Get the original aspect ratio of the earring image
        orig_height, orig_width = earring_image.shape[:2]
        orig_aspect_ratio = orig_width / orig_height

        # Earring categorization
        if earring_type == 'small':
            size_factor = 0.8
            vertical_offset = -10 + y_adjustment
        elif earring_type == 'medium':
            size_factor = 1.0
            vertical_offset = -5 + y_adjustment
        elif earring_type == 'large':
            size_factor = 1.5
            vertical_offset = 10 + y_adjustment
        else:
            size_factor = 1.0
            vertical_offset = 0 + y_adjustment

        # Helper function to apply earring to each ear
        def apply_earring(ear_landmark, is_left_ear):
            h_offset = -horizontal_offset if is_left_ear else horizontal_offset
            dynamic_size_factor = base_size_factor * distance_scale * size_factor

            # Calculate ear region size and position
            ear_x_px = int(ear_landmark.x * frame_width)
            ear_y_px = int(ear_landmark.y * frame_height)

            # Smooth coordinates using Kalman filter
            if is_left_ear:
                ear_x_px, ear_y_px = kalman_left.update(ear_x_px, ear_y_px)
            else:
                ear_x_px, ear_y_px = kalman_right.update(ear_x_px, ear_y_px)

            # Scale the base bounding box size
            base_bbox_size = 15
            scaled_bbox_size = int(base_bbox_size * dynamic_size_factor)

            # Adjust position based on head turn
            turn_offset = int(150 * abs(head_turn_angle) / 90) if (is_left_ear and head_turn_angle > 5) or (not is_left_ear and head_turn_angle < -5) else 0

            if is_left_ear:
                ear_top_left = (ear_x_px - 7 - scaled_bbox_size - turn_offset + h_offset - 5,
                               ear_y_px + 5 - scaled_bbox_size)
                ear_bottom_right = (ear_x_px - 7 + scaled_bbox_size - turn_offset + h_offset - 5,
                                   ear_y_px + 5 + scaled_bbox_size)
            else:
                ear_top_left = (ear_x_px + 7 - scaled_bbox_size + turn_offset + h_offset + 5,
                               ear_y_px + 5 - scaled_bbox_size)
                ear_bottom_right = (ear_x_px + 7 + scaled_bbox_size + turn_offset + h_offset + 5,
                                   ear_y_px + 5 + scaled_bbox_size)

            # Ensure the region is within frame bounds
            ear_top_left = (max(0, int(ear_top_left[0])), max(0, int(ear_top_left[1])))
            ear_bottom_right = (min(frame_width, int(ear_bottom_right[0])),
                               min(frame_height, int(ear_bottom_right[1])))

            # Recalculate dimensions after boundary adjustment
            ear_width = ear_bottom_right[0] - ear_top_left[0]
            ear_height = ear_bottom_right[1] - ear_top_left[1]

            if ear_width <= 0 or ear_height <= 0:
                return None

            # Resize earring while maintaining the original aspect ratio
            target_height = ear_height
            target_width = int(target_height * orig_aspect_ratio)

            if target_width > ear_width:
                target_width = ear_width
                target_height = int(target_width / orig_aspect_ratio)

            resized_earring = cv2.resize(earring_image, (target_width, target_height))

            # Create a new image with the same dimensions as the ear region
            final_earring = np.zeros((ear_height, ear_width, 4), dtype=np.uint8)

            # Position the resized earring at the center of the ear region
            x_offset = (ear_width - target_width) // 2
            y_offset = (ear_height - target_height) // 2

            final_earring[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = resized_earring

            # Apply the mask for transparency
            alpha_channel = final_earring[:, :, 3]
            mask = alpha_channel[:, :, np.newaxis] / 255.0
            overlay = final_earring[:, :, :3] * mask

            region = frame[ear_top_left[1]:ear_bottom_right[1], ear_top_left[0]:ear_bottom_right[0]]
            mask_inv = 1 - mask
            region_inv = region * mask_inv
            combined = cv2.add(overlay.astype(np.uint8), region_inv.astype(np.uint8))

            frame[ear_top_left[1]:ear_bottom_right[1], ear_top_left[0]:ear_bottom_right[0]] = combined

            return {
                'x': (ear_top_left[0] + ear_width / 2) / frame_width,
                'y': (ear_top_left[1] + ear_height / 2) / frame_height
            }

        # Apply the earring to the left ear
        if head_turn_angle >= -5:
            left_earring_coords = apply_earring(left_ear_landmark, is_left_ear=True)
            if left_earring_coords:
                left_ear_detected = True

        # Apply the earring to the right ear
        if head_turn_angle <= 5:
            right_earring_coords = apply_earring(right_ear_landmark, is_left_ear=False)
            if right_earring_coords:
                right_ear_detected = True

        # Combine coordinates if both ears are detected
        if left_ear_detected and right_ear_detected:
            jewelry_coords = {
                'x': (left_earring_coords['x'] + right_earring_coords['x']) / 2,
                'y': (left_earring_coords['y'] + right_earring_coords['y']) / 2
            }
        elif left_ear_detected:
            jewelry_coords = left_earring_coords
        elif right_ear_detected:
            jewelry_coords = right_earring_coords

    return frame, jewelry_coords
