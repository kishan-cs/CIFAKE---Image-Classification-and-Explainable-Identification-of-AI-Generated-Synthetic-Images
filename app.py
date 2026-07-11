#app.py
from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from gradcam_utils import make_gradcam_heatmap_and_time, save_gradcam
from ultralytics import YOLO
import time
import uuid
from functools import wraps

# --- CONFIGURATION ---
IMAGE_SIZE = 128
MODEL_PATH = "cifake_model.h5"
HISTORY_LIMIT = 10
USERS_FILE = "users.json"  # Persistent user storage
# ---------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'))
app.secret_key = 'your_super_secret_key_change_me'  # MUST be set for sessions to work!

# --- Persistent User Storage ---
def load_users():
    """Load users from JSON file. Creates file with default user if not found."""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    # Default demo user — saved to file on first run
    default = {"testuser": "password123"}
    save_users(default)
    return default

def save_users(users_dict):
    """Save users dict to JSON file."""
    with open(USERS_FILE, 'w') as f:
        json.dump(users_dict, f, indent=2)

# Load users at startup
USERS = load_users()
# --------------------------------

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found. Please run main.py first.")
    exit()

# --- Load Models on Startup ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("CIFAKE model loaded successfully.")

    # Dummy prediction to fully build the model graph
    dummy_input = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    model.predict(dummy_input, verbose=0)

    yolo_model = YOLO('yolov8n.pt')
    print("YOLOv8 object detection model loaded.")

except Exception as e:
    print(f"Error during model loading: {e}")
    exit()
# ----------------------------------------------------

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "static", "gradcam")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

# --- Login Required Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            flash("Please log in to access the detector.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function
# ------------------------------------

@app.before_request
def initialize_session():
    """Initialize history if it doesn't exist."""
    if 'history' not in session:
        session['history'] = []

# --- LOGIN / REGISTER / LOGOUT ROUTES ---

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        # Re-load users fresh from file to catch any newly registered users
        users = load_users()
        if username in users and users[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template("login.html", error="Invalid Username or Password.")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        if not username or not password:
            return render_template("register.html", error="Both fields are required.")

        # Re-load fresh to avoid race conditions
        users = load_users()

        if username in users:
            return render_template("register.html", error="Username already exists.")

        # Save new user persistently to file
        users[username] = password
        save_users(users)

        session['logged_in'] = True
        session['username'] = username
        return redirect(url_for('index'))
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    session.pop('history', None)
    return redirect(url_for('login'))

# --- MAIN ROUTES (PROTECTED) ---

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != '':
            original_filename = file.filename
            file_extension = os.path.splitext(original_filename)[1]
            filename = str(uuid.uuid4()) + file_extension
            new_filepath = os.path.join(UPLOAD_FOLDER, filename)

            file.save(new_filepath)

            # 1. Object Detection
            yolo_results = yolo_model.predict(source=new_filepath, verbose=False)
            detected_objects = [yolo_model.names[int(c)] for r in yolo_results for c in r.boxes.cls]

            if detected_objects:
                object_text = f"Image represents: {', '.join(list(set(detected_objects))).upper()}"
            else:
                object_text = "No prominent objects detected."

            # 2. CIFAKE Prediction + Grad-CAM
            try:
                img = Image.open(new_filepath).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            except Exception as e:
                os.remove(new_filepath)
                return render_template("index.html", error=f"Could not process image: {e}")

            img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

            heatmap, inference_time, prediction_prob = make_gradcam_heatmap_and_time(img_array, model)

            if prediction_prob >= 0.5:
                confidence = prediction_prob * 100
                result = "🟢 REAL IMAGE"
                map_explanation = "The heatmap highlights areas the model analyzed to **confirm the image's realism** (e.g., natural textures, smooth edges)."
            else:
                confidence = (1 - prediction_prob) * 100
                result = "🔴 FAKE (AI-generated)"
                map_explanation = "The heatmap highlights artificial patterns, strange edges, or pixel anomalies that led the model to conclude the image is **AI-generated**."

            # 3. Save Grad-CAM
            gradcam_filename = "gradcam_result_" + filename
            gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_filename)
            save_gradcam(new_filepath, heatmap, gradcam_path, target_size=IMAGE_SIZE)

            # 4. Update History
            session['history'].insert(0, {
                'filename': filename,
                'result': result,
                'confidence': f"{confidence:.2f}%",
                'time': f"{inference_time:.2f} ms"
            })
            session['history'] = session['history'][:HISTORY_LIMIT]
            session.modified = True

            # 5. Render result
            return render_template("result.html",
                                   result=result,
                                   confidence_level=f"{confidence:.2f}",
                                   inference_time=f"{inference_time:.2f}",
                                   object_description=object_text,
                                   map_explanation=map_explanation,
                                   uploaded_image=filename,
                                   gradcam_image=gradcam_filename,
                                   history=session['history'],
                                   username=session.get('username'))

        return render_template("index.html", error="Please select a file to upload.",
                               username=session.get('username'), history=session['history'])

    return render_template("index.html", username=session.get('username'), history=session['history'])


@app.route('/clear_history')
@login_required
def clear_history():
    session.pop('history', None)
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)