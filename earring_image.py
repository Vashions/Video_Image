import os
import cv2
import numpy as np
import mediapipe as mp

# Updated advanced necklace extraction to handle transparency better


def earring_process_images(customer_image, earring_image, jewelry_path):
    customer_image1 = cv2.imdecode(np.frombuffer(customer_image, np.uint8), cv2.IMREAD_COLOR)
    earring_image1 = cv2.imdecode(np.frombuffer(earring_image, np.uint8), cv2.IMREAD_UNCHANGED)

    # Remove background of earring image
    earring_image1 = remove_background(earring_image1)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    img_rgb = cv2.cvtColor(customer_image1, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(img_rgb)

    jewelry_filename = os.path.basename(jewelry_path).lower()
    size_factor = 0.25  # Default smaller size factor for earrings

    if result.multi_face_landmarks:
        ih, iw, _ = customer_image1.shape
        
        for face_landmarks in result.multi_face_landmarks:
            for point_index, is_left_ear in [(177, True), (401, False)]:
                point = tuple(map(int, (face_landmarks.landmark[point_index].x * iw, 
                                      face_landmarks.landmark[point_index].y * ih)))

                face_width = int(abs(face_landmarks.landmark[454].x - face_landmarks.landmark[234].x) * iw)
                earring_height = int(face_width * size_factor)
                aspect_ratio = earring_image1.shape[1] / earring_image1.shape[0]
                earring_width = int(earring_height * aspect_ratio)

                x_offset = -earring_width if is_left_ear else 0
                y_offset = -earring_height // 2 - 10
                center_x = point[0] + x_offset
                center_y = point[1] + y_offset

                center_x = max(0, min(center_x, iw - earring_width))
                center_y = max(0, min(center_y, ih - earring_height))

                resized_earring = cv2.resize(earring_image1, (earring_width, earring_height))

                if resized_earring.shape[2] == 4:
                    alpha_channel = resized_earring[:, :, 3]
                    mask = alpha_channel[:, :, np.newaxis] / 255.0
                else:
                    mask = np.ones_like(resized_earring[:, :, :3])

                mask = np.dstack([mask] * 3)
                roi = customer_image1[center_y:center_y+earring_height, center_x:center_x+earring_width]

                if roi.shape[:2] != resized_earring[:, :, :3].shape[:2]:
                    continue

                blended = (mask * resized_earring[:, :, :3] + (1 - mask) * roi).astype(np.uint8)
                customer_image1[center_y:center_y+earring_height, center_x:center_x+earring_width] = blended

    return customer_image1

def remove_background(image):
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    mask = np.all(image[:, :, :3] > [240, 240, 240], axis=2)
    image[mask] = [0, 0, 0, 0]
    
    return image
