import os
import cv2
import numpy as np
import mediapipe as mp

# Define the function to process ring images
def ring_process_images(customer_image, ring_image, jewelry_path):
    customer_image1 = cv2.imdecode(np.frombuffer(customer_image, np.uint8), cv2.IMREAD_COLOR)
    ring_image1 = cv2.imdecode(np.frombuffer(ring_image, np.uint8), cv2.IMREAD_UNCHANGED)

    # Remove background of the ring image
    ring_image1 = remove_background(ring_image1)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

    img_rgb = cv2.cvtColor(customer_image1, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        ih, iw, _ = customer_image1.shape

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
            aspect_ratio = ring_image1.shape[1] / ring_image1.shape[0]
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
            resized_ring = cv2.resize(ring_image1, (ring_width, ring_height), interpolation=cv2.INTER_AREA)

            # Overlay ring image
            if resized_ring.shape[2] == 4:  # If the ring has an alpha channel
                alpha_channel = resized_ring[:, :, 3] / 255.0
                mask = np.dstack([alpha_channel] * 3)
                region = customer_image1[start_y:start_y + ring_height, start_x:start_x + ring_width]

                # Ensure the region matches the resized ring dimensions
                if region.shape[:2]!= resized_ring[:, :, :3].shape[:2]:
                    continue

                blended = (mask * resized_ring[:, :, :3] + (1 - mask) * region).astype(np.uint8)
                customer_image1[start_y:start_y + ring_height, start_x:start_x + ring_width] = blended
            else:
                customer_image1[start_y:start_y + ring_height, start_x:start_x + ring_width] = resized_ring

    return customer_image1


def remove_background(image):
    # Convert image to RGBA if it's not already
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Create a binary mask of non-white pixels
    mask = np.all(image[:, :, :3] < [250, 250, 250], axis=2)

    # Apply the mask to the alpha channel
    image[:, :, 3] = mask.astype(np.uint8) * 255

    return image
