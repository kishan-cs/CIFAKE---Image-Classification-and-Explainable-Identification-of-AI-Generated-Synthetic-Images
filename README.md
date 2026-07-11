# CIFAKE---Image-Classification-and-Explainable-Identification-of-AI-Generated-Synthetic-Images
Classify, Explain, and Visualize — An AI-powered forensic tool that detects whether an image is real or synthetically generated, with full Grad-CAM explainability.
📌 Overview

CIFAKE is a web-based AI image forensics application built with Python Flask and a custom-trained CNN (Convolutional Neural Network). It analyzes uploaded images and determines whether they are real photographs or AI-generated synthetic images, while visually explaining the model's decision using Grad-CAM heatmaps.

This project was built as a standalone deep learning tool focused on making AI predictions transparent and interpretable — not just a black box verdict.


🚀 Features


🔍 Real vs Fake Detection — Binary classification using a custom-trained CNN model
🧠 Grad-CAM Explainability — Visual heatmap overlay showing exactly which regions of the image influenced the model's decision
📦 YOLOv8 Object Detection — Automatically identifies and labels prominent objects in the uploaded image
📊 Confidence Scoring — Displays prediction confidence as a percentage
⚡ Inference Time Tracking — Shows model response time in milliseconds
🕓 Prediction History — Tracks last 10 analyses per session with thumbnails, results, confidence, and timing
🔐 User Authentication — Login and registration system with persistent user storage via users.json
🎨 Professional Dark UI — Built with custom CSS using Syne, Space Mono, and Inter fonts



🛠️ Tech Stack

LayerTechnologyBackendPython, FlaskDeep LearningTensorFlow / Keras (CNN)ExplainabilityGrad-CAM (custom implementation)Object DetectionYOLOv8 (Ultralytics)Image ProcessingOpenCV, PillowFrontendHTML5, CSS3 (custom design system)StorageJSON (users + session history)


📁 Project Structure

cifake/
├── app.py                  # Main Flask application
├── main.py                 # CNN model training script
├── gradcam_utils.py        # Grad-CAM heatmap generation utilities
├── cifake_model.h5         # Trained CNN model (generated after training)
├── users.json              # Persistent user credentials storage
├── static/
│   ├── style.css           # Complete design system
│   ├── uploads/            # User-uploaded images
│   └── gradcam/            # Generated Grad-CAM overlays
├── templates/
│   ├── login.html          # Login page
│   ├── register.html       # Registration page
│   ├── index.html          # Upload / home page
│   └── result.html         # Analysis result page
└── synthetic_images/       # Training dataset (REAL / FAKE folders)


⚙️ Setup & Installation

1. Clone the repository

bashgit clone https://github.com/your-username/cifake.git
cd cifake

2. Install dependencies

bashpip install flask tensorflow opencv-python numpy ultralytics Pillow scikit-learn matplotlib

3. Train the model

Place your dataset in the following structure:

synthetic_images/
└── Train/
    ├── REAL/   ← real image files
    └── FAKE/   ← AI-generated image files

Then run:

bashpython main.py

This generates cifake_model.h5.

4. Run the app

bashpython app.py

Visit http://127.0.0.1:5000 in your browser.

🔐 Default Login

FieldValueUsernametestuserPasswordpassword123

New users can register directly from the app. Credentials persist across restarts via users.json.

🧠 How It Works

User uploads an image via the web interface
YOLOv8 scans the image and identifies visible objects
The image is preprocessed and passed through the CNN classifier
The model outputs a probability score → classified as REAL or FAKE
Grad-CAM generates a heatmap showing the model's areas of focus
Results (verdict, confidence, heatmap, inference time) are displayed and saved to session history

📸 Grad-CAM Explained

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions of an image that most influenced the model's decision:

🔴 Red/Yellow areas → High attention — model focused here most
🔵 Blue/Cool areas → Low attention — model largely ignored these

For real images, highlights typically appear on natural textures and smooth edges.
For AI-generated images, highlights often appear on unnatural artifacts, blurry boundaries, or inconsistent pixel patterns.

📌 Notes

The model is trained on the CIFAKE dataset (or equivalent real/fake image pairs)
cifake_model.h5 is excluded from the repository — run main.py to generate it
For phone/device testing over a local network, use ngrok

The model is trained on the CIFAKE dataset (or equivalent real/fake image pairs)
cifake_model.h5 is excluded from the repository — run main.py to generate it
For phone/device testing over a local network, use ngrok
