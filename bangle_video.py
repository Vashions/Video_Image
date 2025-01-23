import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def rotate_image_with_padding(image, angle, padding_factor=1.5):
    """Rotate an image with increased canvas size to prevent clipping."""
    (h, w) = image.shape[:2]
    canvas_size = int(max(h, w) * padding_factor)
    padded_image = np.zeros((canvas_size, canvas_size, 4), dtype=image.dtype)
    
    # Place the original image at the center of the padded canvas
    offset = (canvas_size - w) // 2
    padded_image[offset:offset + h, offset:offset + w] = image

    # Rotate the padded image
    center = (canvas_size // 2, canvas_size // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        padded_image, matrix, (canvas_size, canvas_size), 
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return rotated

def overlay_image(background, overlay, x, y):
    """Overlay an image onto a background at (x, y) coordinates."""
    h, w = overlay.shape[:2]
    x_end, y_end = x + w, y + h
    if x < 0 or y < 0 or x_end > background.shape[1] or y_end > background.shape[0]:
        return background  # Skip overlay if out of bounds

    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(0, 3):
        background[y:y_end, x:x_end, c] = (
            alpha_overlay * overlay[:, :, c] +
            alpha_background * background[y:y_end, x:x_end, c]
        )
    return background

def process_left_bangle(frame, bangle_image, wrist, angle, scaling_factor, x_offset_left, y_offset_left, left_initial_angle_offset):
    """Process and overlay the left bangle."""
    wrist_x, wrist_y = int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])

    adjusted_angle = angle + left_initial_angle_offset
    # Compute dynamic vertical offset adjustment based on rotation angle
    dynamic_offset = int(np.abs(angle) * 0.2)  # Scale the upward movement proportionally to rotation

    # Resize and rotate the bangle
    bangle_size = int(100 * scaling_factor)
    bangle_resized = cv2.resize(bangle_image, (bangle_size, bangle_size), interpolation=cv2.INTER_AREA)
    rotated_bangle = rotate_image_with_padding(bangle_resized, adjusted_angle)

    # Calculate bangle position based on wrist and offsets
    x = wrist_x - rotated_bangle.shape[1] // 2 + x_offset_left
    y = wrist_y - rotated_bangle.shape[0] // 2 + y_offset_left - dynamic_offset

    # Check for boundary conditions for left bangle
    if x < 0 or y < 0 or x + rotated_bangle.shape[1] > frame.shape[1] or y + rotated_bangle.shape[0] > frame.shape[0]:
        print(f"Left Bangle is out of frame boundaries.")
        return None  # Skip overlay if out of bounds
    
    return (x, y, rotated_bangle)

def process_bangle_design(
    frame, bangle_image, scaling_factor=1.1,
    x_offset_left=30, y_offset_left=70, left_initial_angle_offset=-5
):
    if bangle_image is None:
        print("Bangle image is None.")
        return frame

    # Convert frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Initialize variable to store bangle position for left hand
    bangle_left_position = None

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # If right hand is detected, skip the bangle processing and stop
            if hand_idx == 1:  # Right hand detected
                print("Right hand detected, skipping bangle overlay.")
                return frame

            # Get the wrist landmark (Left hand)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            wrist_x, wrist_y = int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])

            # Get the wrist rotation angle using the vector between WRIST and INDEX_FINGER_MCP
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            index_mcp_x, index_mcp_y = int(index_mcp.x * frame.shape[1]), int(index_mcp.y * frame.shape[0])
            angle = -np.degrees(np.arctan2(index_mcp_y - wrist_y, index_mcp_x - wrist_x)) - 90

            # Process the left bangle
            bangle_left_position = process_left_bangle(frame, bangle_image, wrist, angle, scaling_factor, x_offset_left, y_offset_left, left_initial_angle_offset)

        # Overlay the left bangle if the position is calculated
        if bangle_left_position:
            x_left, y_left, rotated_bangle_left = bangle_left_position
            frame = overlay_image(frame, rotated_bangle_left, x_left, y_left)

    return frame
