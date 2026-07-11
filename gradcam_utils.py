# gradcam_utils.py
import numpy as np
import tensorflow as tf
import cv2
import os
import time
# Set target size to match the model
TARGET_SIZE = 128
LAST_CONV_LAYER_NAME = "final_conv"

# NOTE: img_path is removed from this signature to match the call in app.py
def make_gradcam_heatmap_and_time(img_array, model, last_conv_layer_name=LAST_CONV_LAYER_NAME):
   """
   Generates Grad-CAM heatmap and calculates prediction inference time.
   Returns: heatmap, inference_time_ms, prediction_prob
   """
   # 1. Prediction and Timing
   start_time = time.time()
   prediction_prob = model.predict(img_array, verbose=0)[0][0]
   inference_time = (time.time() - start_time) * 1000 # Time in milliseconds

   # 2. Grad-CAM Calculation
   grad_model = tf.keras.models.Model(
       inputs=model.layers[0].input,
       outputs=[model.get_layer(last_conv_layer_name).output, model.layers[-1].output]
   )

   with tf.GradientTape() as tape:
       conv_outputs, preds = grad_model(img_array)
       loss = preds[0]

   grads = tape.gradient(loss, conv_outputs)

   if grads is None:
       print("Warning: Gradients are None. Returning blank heatmap.")
       heatmap = np.zeros(conv_outputs.shape[1:3], dtype=np.float32)
   else:
       pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

       conv_outputs = conv_outputs[0]
       heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)

       heatmap = heatmap.numpy()
       heatmap = np.maximum(heatmap, 0)
       max_val = np.max(heatmap)
       if max_val != 0:
           heatmap /= max_val

   return heatmap, inference_time, prediction_prob

def save_gradcam(img_path, heatmap, cam_path="gradcam_output.jpg", alpha=0.4, target_size=TARGET_SIZE):
   """Saves the Grad-CAM heatmap overlay on the original image."""
   img = cv2.imread(img_path)
   img = cv2.resize(img, (target_size, target_size))

   if len(img.shape) < 3 or img.shape[2] == 1:
       img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
   elif img.shape[2] == 4:
       img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

   heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
   heatmap = np.uint8(255 * heatmap)

   jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
   superimposed_img = cv2.addWeighted(img, 1 - alpha, jet_heatmap, alpha, 0)

   cv2.imwrite(cam_path, superimposed_img)

   return cam_path