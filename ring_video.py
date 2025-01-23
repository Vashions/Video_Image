import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Track only one hand
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Kalman filter for smoothing positions
position_kalman = cv2.KalmanFilter(4, 2)
position_kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
position_kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
position_kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01  # Reduced process noise
position_kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.05  # Increased measurement noise for stability
position_kalman.errorCovPost = np.eye(4, dtype=np.float32)
position_kalman.statePost = np.zeros((4, 1), dtype=np.float32)

# Initialize Kalman filter for smoothing ring size
size_kalman = cv2.KalmanFilter(2, 1)
size_kalman.measurementMatrix = np.array([[1, 0]], np.float32)
size_kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
size_kalman.processNoiseCov = np.eye(2, dtype=np.float32) * 0.005  # Reduced process noise
size_kalman.measurementNoiseCov = np.array([[0.05]], np.float32)  # Increased measurement noise for stability
size_kalman.errorCovPost = np.eye(2, dtype=np.float32)
size_kalman.statePost = np.array([[50], [0]], np.float32)  # Initial ring size

# Helper function to rotate an image
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_rgb = cv2.warpAffine(image[:, :, :3], rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rotated_alpha = cv2.warpAffine(image[:, :, 3], rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return cv2.merge([rotated_rgb, rotated_alpha])

# Helper function to overlay an image with transparency
def overlay_image(background, overlay, x, y):
    if overlay.shape[2] == 4:  # If overlay has an alpha channel
        alpha = overlay[:, :, 3] / 255.0  # Normalize alpha channel
        for c in range(3):  # Loop over color channels
            background[y:y+overlay.shape[0], x:x+overlay.shape[1], c] = \
                (1 - alpha) * background[y:y+overlay.shape[0], x:x+overlay.shape[1], c] + \
                alpha * overlay[:, :, c]
    else:
        background[y:y+overlay.shape[0], x:x+overlay.shape[1]] = overlay
    return background


# Define smoothing variables outside the function (global)
center_x_smooth, center_y_smooth = None, None

def process_ring_design(frame, ring_image, scaling_factor=0.5):
    global center_x_smooth, center_y_smooth  # Access global smoothing variables

    if ring_image is None:
        return frame, None

    # Convert frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Create a copy of frame for processing
    processed_frame = frame.copy()

    # Set faster smoothing parameters
    smoothing_alpha = 0.6  # Lower smoothing factor for faster response

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get key landmarks for ring placement
            mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]

            # Convert normalized coordinates to pixel values
            mcp_x, mcp_y = int(mcp.x * frame.shape[1]), int(mcp.y * frame.shape[0])
            pip_x, pip_y = int(pip.x * frame.shape[1]), int(pip.y * frame.shape[0])

            # Apply Kalman filter for smoothing position
            measured_position = np.array([[np.float32((mcp_x + pip_x) / 2)], [np.float32((mcp_y + pip_y) / 2)]], np.float32)
            position_kalman.correct(measured_position)
            predicted_position = position_kalman.predict()

            predicted_center_x, predicted_center_y = int(predicted_position[0]), int(predicted_position[1])

            # Apply exponential smoothing to stabilize and speed up position updates
            if center_x_smooth is None:
                center_x_smooth, center_y_smooth = predicted_center_x, predicted_center_y
            else:
                center_x_smooth = smoothing_alpha * center_x_smooth + (1 - smoothing_alpha) * predicted_center_x
                center_y_smooth = smoothing_alpha * center_y_smooth + (1 - smoothing_alpha) * predicted_center_y

            center_x, center_y = int(center_x_smooth), int(center_y_smooth)

            # Calculate distance between MCP and PIP for resizing
            distance = np.sqrt((pip_x - mcp_x) ** 2 + (pip_y - mcp_y) ** 2)
            measured_size = np.array([[np.float32(distance * scaling_factor)]], np.float32)
            size_kalman.correct(measured_size)
            predicted_size = size_kalman.predict()
            ring_size = max(20, min(100, int(predicted_size[0])))  # Constrain ring size

            # Calculate rotation angle
            angle = -np.degrees(np.arctan2(pip_y - mcp_y, pip_x - mcp_x)) - 90

            # Ensure ring image has alpha channel
            if ring_image.shape[2] == 3:
                ring_image = cv2.cvtColor(ring_image, cv2.COLOR_BGR2BGRA)

            # Resize and rotate ring image
            ring_resized = cv2.resize(ring_image, (ring_size, ring_size), interpolation=cv2.INTER_AREA)
            rotated_ring = rotate_image(ring_resized, angle)

            # Calculate overlay region
            x1 = center_x - ring_size // 2
            y1 = center_y - ring_size // 2

            # Check frame boundaries
            if (x1 >= 0 and y1 >= 0 and 
                x1 + ring_size <= frame.shape[1] and 
                y1 + ring_size <= frame.shape[0]):
                # Overlay ring using the overlay_image function
                processed_frame = overlay_image(processed_frame, rotated_ring, x1, y1)

    return processed_frame, results
