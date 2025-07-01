"""
Flask Web Application for Drone Detection

This script runs a web server that allows users to upload an image.
The server processes the image using a pre-trained Faster R-CNN model
and displays the result with detected drones highlighted by bounding boxes.
"""
import os
import uuid
import logging
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import torch
from PIL import Image

# Local module imports
from inference import AppConfig, create_detection_model, run_prediction

# --- App Configuration Class ---
class Config:
    """Flask configuration variables."""
    STATIC_FOLDER = 'static'
    UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
    PREDICTION_FOLDER = os.path.join(STATIC_FOLDER, 'predictions')
    MODEL_PATH = "fasterrcnn_drone_detector.pth"
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit for uploads

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREDICTION_FOLDER'], exist_ok=True)

# --- Model Loading ---
# Load the configuration and device
config = AppConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app.logger.info(f"Running on device: {device}")

# Load the model structure and weights
model_path = app.config['MODEL_PATH']
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please place your trained model in the root directory.")

model = create_detection_model(num_classes=config.NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
app.logger.info("Model loaded successfully.")

# --- Helper Function ---
def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')

    # Handle POST request
    if 'file' not in request.files:
        return render_template('index.html', error='No file part in the request.')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error='No file selected.')

    if not allowed_file(file.filename):
        return render_template('index.html', error='File type not allowed. Please upload a JPG, JPEG, or PNG.')

    try:
        # Securely save the uploaded file with a unique name to prevent overwrites
        original_filename = secure_filename(file.filename)
        file_ext = os.path.splitext(original_filename)[1]
        unique_upload_filename = f"{uuid.uuid4()}{file_ext}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_upload_filename)
        file.save(upload_path)
        
        # Run prediction
        result_image_np = run_prediction(upload_path, model, device, config)
        
        result_image = Image.fromarray(result_image_np)
        prediction_filename = f"{uuid.uuid4()}.jpg"
        prediction_path = os.path.join(app.config['PREDICTION_FOLDER'], prediction_filename)
        result_image.save(prediction_path)
        
        return render_template(
            'index.html',
            original_filename=unique_upload_filename,
            prediction_filename=prediction_filename
        )
    except Exception as e:
        app.logger.error(f"An error occurred during processing: {e}", exc_info=True)
        return render_template('index.html', error='Failed to process the image.')

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files from the UPLOAD_FOLDER."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predictions/<path:filename>')
def predicted_file(filename):
    """Serve prediction result files from the PREDICTION_FOLDER."""
    return send_from_directory(app.config['PREDICTION_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)