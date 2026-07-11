# main.py (full)
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time # New import for timing

# --- CONFIGURATION ---
IMAGE_SIZE = 128
# ---

# Load Images
def load_images(folder, label):
   images, labels = [], []
   for file in os.listdir(folder):
       path = os.path.join(folder, file)
       try:
           img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
           images.append(np.array(img, dtype=np.float32) / 255.0)
           labels.append(label)
       except:
           continue
   return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)

print("Loading datasets...")
x_real, y_real = load_images("synthetic_images/Train/REAL", 1)
x_fake, y_fake = load_images("synthetic_images/Train/FAKE", 0)

x = np.concatenate([x_real, x_fake], axis=0)
y = np.concatenate([y_real, y_fake], axis=0)

# Split data
x_train, x_val, y_train, y_val = train_test_split(
   x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)


train_datagen = ImageDataGenerator(
   rotation_range=20,
   width_shift_range=0.1,
   height_shift_range=0.1,
   shear_range=0.1,
   zoom_range=0.1,
   horizontal_flip=True,
   fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(x_train, y_train, batch_size=16)
val_generator = val_datagen.flow(x_val, y_val, batch_size=16)

# Simple CNN Model (with named final conv layer)
print("Building model...")
model = Sequential([
   tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
   Conv2D(32, (3,3), activation='relu'),
   MaxPooling2D(2,2),
   Conv2D(64, (3,3), activation='relu'),
   MaxPooling2D(2,2),
   Conv2D(128, (3,3), activation='relu', name='final_conv'), # Layer name changed/added for robustness
   MaxPooling2D(2,2),
   Flatten(),
   Dense(64, activation='relu'),
   Dropout(0.5),
   Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train with increased epochs and data generators
print("Training model...")
model.fit(
   train_generator,
   epochs=20,  # Increased epochs for better training
   validation_data=val_generator,
   verbose=1
)

# Save Model
model.save("cifake_model.h5")
print("Model saved as cifake_model.h5")

# Grad-CAM Function
def get_gradcam(model, img_path, layer_name="final_conv"):
   """
   Generates and saves Grad-CAM images for a given input image.
   """
   img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
   x = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

   start_time = time.time() # Start timing prediction
   pred_prob = model.predict(x, verbose=0)[0][0]
   inference_time = (time.time() - start_time) * 1000 # End timing and convert to ms

   pred_class = 1 if pred_prob > 0.5 else 0

   target_layer = model.get_layer(layer_name)

   grad_model = tf.keras.models.Model(
       inputs=model.layers[0].input,
       outputs=[target_layer.output, model.layers[-1].output]
   )

   with tf.GradientTape() as tape:
       conv_outputs, predictions = grad_model(x)
       loss = predictions[0]

   grads = tape.gradient(loss, conv_outputs)
   pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
   conv_outputs = conv_outputs[0]
   heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)

   heatmap = np.maximum(heatmap.numpy(), 0)
   if np.max(heatmap) != 0:
       heatmap = heatmap / np.max(heatmap)

   heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
   heatmap = np.uint8(255 * heatmap)

   heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
   original_img = cv2.imread(img_path)
   original_img = cv2.resize(original_img, (IMAGE_SIZE, IMAGE_SIZE))

   if original_img.shape[2] == 4:
       original_img = cv2.cvtColor(original_img, cv2.COLOR_BGRA2BGR)
   elif len(original_img.shape) < 3 or original_img.shape[2] == 1:
       original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

   superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

   os.makedirs("gradcam_output", exist_ok=True)

   cv2.imwrite("gradcam_output/original.jpg", original_img)
   cv2.imwrite("gradcam_output/heatmap.jpg", heatmap_colored)
   cv2.imwrite("gradcam_output/gradcam_result.jpg", superimposed_img)

   return pred_class, superimposed_img, pred_prob, inference_time

#-----------------------testing image--------------------------------------
test_image = "synthetic_images/Test/FAKE/WIN_20260109_12_11_53_Pro.jpg"
if os.path.exists(test_image):

   pred_class, gradcam_image, pred_prob, inference_time = get_gradcam(model, test_image)

   if pred_prob > 0.5:
       confidence = pred_prob * 100
       result = "🟢 REAL IMAGE"
       map_explanation = "The heatmap highlights areas the model analyzed to confirm the image's realism (e.g., natural textures, smooth edges)."
   else:
       confidence = (1 - pred_prob) * 100
       result = "🔴 FAKE IMAGE"
       map_explanation = "The heatmap highlights artificial patterns, strange edges, or pixel anomalies that led the model to conclude the image is AI-generated."

   print(f"Predicted: {result}")
   print(f"Confidence Level: {confidence:.2f}%")
   print(f"Inference Time: {inference_time:.2f} ms") # New result
   print(f"Confidence Map Explanation: {map_explanation}") # New result

   print("Grad-CAM results saved in gradcam_output/ folder:")
   print("- original.jpg: Original input image")
   print("- heatmap.jpg: Heatmap visualization")
   print("- gradcam_result.jpg: Superimposed result ")

   plt.imshow(cv2.cvtColor(gradcam_image, cv2.COLOR_BGR2RGB))
   plt.title("Grad-CAM Result")
   plt.axis('off')
   plt.show()
else:
   print(f"image not found: {test_image}")