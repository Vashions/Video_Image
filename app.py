from flask import Flask, flash, request, redirect, render_template, send_file, jsonify, send_from_directory
from necklace_image import necklace_process_images, detect_and_overlay_accessory, allowed_file, earring_process_images
from ring_image import ring_process_images
from earring_video import process_earring_design
from necklace_video import process_necklace_design  
from bangle_video import process_bangle_design
from ring_video import process_ring_design , overlay_image
from werkzeug.utils import secure_filename
from collections import defaultdict
from ultralytics import YOLO
from flask_cors import CORS 
from PIL import Image
from io import BytesIO
import mediapipe as mp
import requests
import numpy as np
import base64
import time
import cv2
import io
import os

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Set app configurations
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

# Load initial jewelry images from respective folders
NecklaceFolderPath = "static/Image/Necklace"
listnecklace = os.listdir(NecklaceFolderPath)
necklace_image = cv2.imread(os.path.join(NecklaceFolderPath, listnecklace[0]), cv2.IMREAD_UNCHANGED)

EarringFolderPath = "static/Image/Earring"
listearring = os.listdir(EarringFolderPath)
earring_image = cv2.imread(os.path.join(EarringFolderPath, listearring[0]), cv2.IMREAD_UNCHANGED)

BangleFolderPath = "static/Image/Bangle"
listbangle = os.listdir(BangleFolderPath)
bangle_image = cv2.imread(os.path.join(BangleFolderPath, listbangle[0]), cv2.IMREAD_UNCHANGED)

RingFolderPath = "static/Image/Ring"
listring = os.listdir(RingFolderPath)
ring_image = cv2.imread(os.path.join(RingFolderPath, listring[0]), cv2.IMREAD_UNCHANGED)

# Dictionary to track currently active jewelry selections
active_jewelry = {
    'necklace': None,
    'earring': None,
    'bangle': None,
    'ring': None
}

def update_active_jewelry(jewelry_type, path):
    """Track active jewelry selections"""
    active_jewelry[jewelry_type] = path
    return active_jewelry

# Routes for serving static files
@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('templates', filename)

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('templates', filename)

# Main route for the application
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image processing
@app.route('/', methods=['POST'])
def process_image():
    # Check if file exists in request
    if 'file' not in request.files:
        flash('No file part')
        return jsonify({"error": "No file part"}), 400
   
    file = request.files['file']
    jewelry_path = request.form.get('jewelry_path')
    jewelry_type = request.form.get('jewelry_type', 'necklace')
   
    # Validate file selection
    if file.filename == '':
        flash('No image selected for uploading')
        return jsonify({"error": "No image selected for uploading"}), 400
   
    if file and allowed_file(file.filename):
        try:
            img_data = file.read()
           
            # Handle jewelry image from URL or local path
            if jewelry_path.startswith('http'):
                response = requests.get(jewelry_path)
                jewelry_image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
                jewelry_filename = os.path.basename(jewelry_path)
            else:
                jewelry_image = cv2.imread(jewelry_path, cv2.IMREAD_UNCHANGED)
                jewelry_filename = os.path.basename(jewelry_path)
           
            if jewelry_image is None:
                flash('Invalid jewelry image')
                return jsonify({"error": "Invalid jewelry image"}), 400
           
            # Apply background removal
            jewelry_image = remove_background(jewelry_image)
           
            # Process image with jewelry overlay
            modified_img = detect_and_overlay_accessory(
                BytesIO(img_data),
                jewelry_image,
                jewelry_type,
                jewelry_filename
            )
           
            # Encode and return processed image
            _, encoded_modified_img = cv2.imencode('.png', modified_img)
            modified_img_base64 = base64.b64encode(encoded_modified_img).decode('utf-8')
           
            return send_file(BytesIO(base64.b64decode(modified_img_base64)), mimetype='image/png')
           
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return jsonify({"error": "Invalid file type"}), 400

# Function to remove background from jewelry images
def remove_background(image):
    if image.shape[2] == 3:
        image = cv2.cvtColor(necklace_image, cv2.COLOR_BGR2BGRA)

    if np.any(image[:,:,3] < 255):
        return image

    # Convert to HSV and create mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 225])
    upper = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.bitwise_not(mask)

    # Apply morphological operations
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Smooth edges
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    # Apply mask to alpha channel
    result = image.copy()
    result[:,:,3] = mask

    return result

# Constants for jewelry folder paths
JewelFolders = {
    'necklace': NecklaceFolderPath,
    'earring': EarringFolderPath,
    'bangle': BangleFolderPath,
    'ring': RingFolderPath
}

# Centralized function for selecting jewelry image
def select_jewelry_image(jewelry_type, jewelry_name):
    folder_path = JewelFolders.get(jewelry_type)
    if not folder_path:
        return {"success": False, "message": f"Invalid jewelry type: {jewelry_type}"}, 400

    jewelry_path = os.path.join(folder_path, jewelry_name)
    if os.path.exists(jewelry_path):
        return {"success": True, "message": f"Selected {jewelry_type}: {jewelry_name}", "path": jewelry_path}, 200
    else:
        return {"success": False, "message": f"{jewelry_type.capitalize()} image not found"}, 404

# Routes for selecting different types of jewelry
@app.route('/select_necklace_image/<necklace_name>', methods=['GET'])
def select_necklace_image(necklace_name):
    response, status = select_jewelry_image('necklace', necklace_name)
    return jsonify(response), status

@app.route('/select_earring_image/<earring_name>', methods=['GET'])
def select_earring_image(earring_name):
    response, status = select_jewelry_image('earring', earring_name)
    return jsonify(response), status

@app.route('/select_bangle_image/<bangle_name>', methods=['GET'])
def select_bangle_image(bangle_name):
    response, status = select_jewelry_image('bangle', bangle_name)
    return jsonify(response), status

@app.route('/select_ring_image/<ring_name>', methods=['GET'])
def select_ring_image(ring_name):
    response, status = select_jewelry_image('ring', ring_name)
    return jsonify(response), status

# Routes for processing different types of jewelry images
@app.route('/necklace_process_image', methods=['POST'])
def necklace_process_images_api():
    try:
        # Read image files
        customer_image = request.files['customer_image'].read()
        image2 = request.files['image2']
        
        jewelry_filename = secure_filename(image2.filename)

        if not jewelry_filename:
            print("No filename received for necklace image.")
            return jsonify({'error': 'No necklace image filename received'}), 400

        image2.seek(0)
        result_image = necklace_process_images(customer_image, image2.read(), jewelry_filename)
        _, buffer = cv2.imencode('.jpg', result_image)

        return send_file(BytesIO(buffer.tobytes()), mimetype='image/jpeg', as_attachment=True, download_name='result.jpg')

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/earring_process_image', methods=['POST'])
def earring_process_images_api():
    try:
        # Verify files exist
        if 'customer_image' not in request.files or 'earring_image' not in request.files:
            return jsonify({'error': 'Missing required image files'})

        # Process images
        customer_image = request.files['customer_image'].read()
        earring_image = request.files['earring_image'].read()
        jewelry_path = request.files['earring_image'].filename

        if not customer_image or not earring_image:
            return jsonify({'error': 'Empty image data received'})

        result_image = earring_process_images(customer_image, earring_image, jewelry_path)

        _, buffer = cv2.imencode('.jpg', result_image)
        return send_file(BytesIO(buffer.tobytes()), mimetype='image/jpeg', as_attachment=True, download_name='result.jpg')
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/ring_process_image', methods=['POST'])
def ring_process_images_api():
    try:
        # Verify files exist
        if 'customer_image' not in request.files or 'ring_image' not in request.files:
            return jsonify({'error': 'Missing required image files'})

        # Process images
        customer_image_file = request.files['customer_image']
        ring_image_file = request.files['ring_image']
        jewelry_path = ring_image_file.filename

        customer_image = customer_image_file.stream.read()
        ring_image = ring_image_file.stream.read()

        if not customer_image or not ring_image:
            return jsonify({'error': 'Empty image data received'})

        result_image = ring_process_images(customer_image, ring_image, jewelry_path)

        _, buffer = cv2.imencode('.jpg', result_image)
        return send_file(BytesIO(buffer.tobytes()), mimetype='image/jpeg', as_attachment=True, download_name='result.jpg')
    except Exception as e:
        return jsonify({'error': str(e)})

# Track selected jewelry types for video processing
selected_jewelry_types = defaultdict(bool)

# Route for processing video frames
@app.route('/process_frame', methods=['POST'])
def process_frame():
    print("Received form data:", request.form)
    print("Received files:", request.files)

    if 'frame' not in request.files:
        return jsonify({"error": "No frame part provided"}), 400

    # Process video frame with selected jewelry
    file = request.files['frame']
    npimg = np.fromfile(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    is_multi_jewel = request.form.get('multi_jewel_enabled', 'false').lower() == 'true'
    processed_frame = frame.copy()
    jewelry_source = 'local'

    try:
        start_time = time.time()
        current_design = request.form.get('design', '').strip()

        # Helper function for overlay
        def overlay_with_alpha(base_image, overlay_image, x, y):
            if overlay_image.shape[2] == 4:
                alpha_mask = overlay_image[:, :, 3] / 255.0
                for c in range(3):
                    base_image[y:y+overlay_image.shape[0], x:x+overlay_image.shape[1], c] = (
                        alpha_mask * overlay_image[:, :, c] +
                        (1 - alpha_mask) * base_image[y:y+overlay_image.shape[0], x:x+overlay_image.shape[1], c]
                    )
            else:
                base_image[y:y+overlay_image.shape[0], x:x+overlay_image.shape[1]] = overlay_image
            return base_image

        # Process multiple jewelry items if enabled
        if is_multi_jewel:
            # Process each jewelry type in sequence
            if 'necklace_path' in request.form:
                necklace_path = request.form.get('necklace_path').strip()
                if necklace_path:
                    necklace_image = cv2.imread(necklace_path, cv2.IMREAD_UNCHANGED)
                    if necklace_image is not None:
                        necklace_type = 'regular'
                        if '_large2' in necklace_path.lower():
                            necklace_type = 'large2'
                        elif '_large' in necklace_path.lower():
                            necklace_type = 'large'
                        elif '_choker' in necklace_path.lower():
                            necklace_type = 'choker'
                        processed_frame, _ = process_necklace_design(
                            processed_frame, necklace_image, necklace_type=necklace_type)
                        print(f"Processed necklace with type: {necklace_type}")

            jewelry_types = ['earring', 'bangle', 'ring']
            for jewelry_type in jewelry_types:
                path_key = f'{jewelry_type}_path'
                if path_key in request.form:
                    jewelry_path = request.form.get(path_key).strip()
                    if jewelry_path:
                        jewelry_image = cv2.imread(jewelry_path, cv2.IMREAD_UNCHANGED)
                        if jewelry_image is not None:
                            if jewelry_type == 'earring':
                                earring_type = 'medium'
                                if '_small' in jewelry_path.lower():
                                    earring_type = 'small'
                                elif '_large' in jewelry_path.lower():
                                    earring_type = 'large'
                                processed_frame, _ = process_earring_design(
                                    processed_frame, jewelry_image, earring_type=earring_type)
                                print(f"Processed earring with type: {earring_type}")
                            elif jewelry_type == 'bangle':
                                processed_frame, _ = process_bangle_design(processed_frame, jewelry_image)
                                print("Processed bangle")
                            elif jewelry_type == 'ring':
                                processed_frame, _ = process_ring_design(processed_frame, jewelry_image)
                                print("Processed ring")

        else:
            design = current_design if current_design else request.form.get('design', '').strip()
            jewelry_path = request.form.get('jewelry_path', '').strip()
            jewelry_url = request.form.get('jewelry_url', '').strip()

            if not jewelry_path and not jewelry_url:
                return jsonify({"error": "No jewelry path or URL provided"}), 400

            if jewelry_url:
                response = requests.get(jewelry_url)
                if response.status_code != 200:
                    raise ValueError(f"Failed to fetch jewelry image from URL: {jewelry_url}")
                jewelry_image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
                jewelry_source = 'client'
            else:
                jewelry_image = cv2.imread(jewelry_path, cv2.IMREAD_UNCHANGED)
                jewelry_source = 'local'

            if jewelry_image is None:
                raise ValueError(f"Failed to load jewelry image from {'URL' if jewelry_url else 'path'}")

            jewelry_filename = os.path.basename(jewelry_path).lower()
            jewelry_type = ''
            if 'Necklace' in design:
                jewelry_type = 'necklace'
            elif 'Earring' in design:
                jewelry_type = 'earring'
            elif 'Bangle' in design:
                jewelry_type = 'bangle'
            elif 'Ring' in design:
                jewelry_type = 'ring'
            else:
                return jsonify({"error": "Invalid or unknown jewelry design"}), 400

            if jewelry_type == 'necklace':
                necklace_type = 'regular'
                if '_large2' in jewelry_filename:
                    necklace_type = 'large2'
                elif '_large' in jewelry_filename:
                    necklace_type = 'large'
                elif '_choker' in jewelry_filename:
                    necklace_type = 'choker'
                processed_frame, _ = process_necklace_design(frame, jewelry_image, necklace_type=necklace_type)
            elif jewelry_type == 'earring':
                earring_type = 'medium'
                if '_small' in jewelry_filename:
                    earring_type = 'small'
                elif '_large' in jewelry_filename:
                    earring_type = 'large'
                processed_frame, _ = process_earring_design(frame, jewelry_image, earring_type=earring_type)
            elif jewelry_type == 'bangle':
                processed_frame, _ = process_bangle_design(frame, jewelry_image)
            elif jewelry_type == 'ring':
                processed_frame, _ = process_ring_design(frame, jewelry_image)

        end_time = time.time()
        processing_time = end_time - start_time

        _, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        response_data = {
            'image': base64.b64encode(buffer).decode('utf-8'),
            'processing_time': processing_time,
            'jewelry_source': jewelry_source,
            'multi_jewel_enabled': is_multi_jewel,
            'current_design': current_design
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        return jsonify({"error": f"Error processing frame: {str(e)}"}), 500


def resize_jewelry_image(jewelry_image, scale_factor=1.3):
    """Resize jewelry image based on the scale factor."""
    height, width = jewelry_image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_jewelry = cv2.resize(jewelry_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    if scale_factor == 1.3:
        print("Resizing jewelry image for large category.")
    else:
        print("Resizing jewelry image for regular category.")
    
    return resized_jewelry


@app.route('/jewelry/<path:filename>')
def serve_jewelry(filename):
    if filename.startswith('Necklace'):
        return send_from_directory(NecklaceFolderPath, filename)
    elif filename.startswith('Earring'):
        return send_from_directory(EarringFolderPath, filename)
    elif filename.startswith('Bangle'):
        return send_from_directory(BangleFolderPath, filename)
    elif filename.startswith('Ring'):
        return send_from_directory(RingFolderPath, filename)
    else:
        return jsonify({"error": "Invalid jewelry type"}), 400


# video code ended


from flask import Response

@app.route('/process_realtime_client_jewelry', methods=['POST'])
def process_realtime_client_jewelry():
    data = request.json
    jewelry_name = data.get('jewelry_name')
    jewelry_type = data.get('jewelry_type')
    client_name = data.get('client_name')
    frame_data = data.get('frame')

    # Construct the path to the client's jewelry folder
    jewelry_path = os.path.join('Clients', client_name, jewelry_type, jewelry_name)

    # Check if the path exists
    if not os.path.exists(jewelry_path):
        return jsonify({"error": f"Jewelry image '{jewelry_name}' not found for client '{client_name}'"}), 404

    # Load the jewelry image
    jewelry_image = cv2.imread(jewelry_path, cv2.IMREAD_UNCHANGED)

    if jewelry_image is None:
        return jsonify({"error": "Failed to load jewelry image"}), 500

    # Decode the base64 frame data
    frame_bytes = base64.b64decode(frame_data)
    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

    # Process the frame with the jewelry image
    if jewelry_type.lower() == 'necklace':
        processed_frame, _ = process_necklace_design(frame, jewelry_image)
    elif jewelry_type.lower() == 'earring':
        processed_frame, _ = process_earring_design(frame, jewelry_image)
    elif jewelry_type.lower() == 'ring':
        processed_frame, _ = process_ring_design(frame, jewelry_image)
    else:
        return jsonify({"error": "Invalid jewelry type"}), 400

    # Encode the processed frame to base64
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')

    # Return the processed frame
    return jsonify({"image": processed_frame_base64})






@app.route('/update_design', methods=['POST'])
def update_design():
    design = request.args.get('design')
    # Implement design update logic here if needed
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=True)


@app.route('/toggle_multi_jewel', methods=['POST'])
def toggle_multi_jewel():
    is_enabled = request.json.get('enabled', False)
    selected_jewelry_types.clear()  # Reset selections on toggle
    return jsonify({"success": True, "multi_jewel_enabled": is_enabled})
