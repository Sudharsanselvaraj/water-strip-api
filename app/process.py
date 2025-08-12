import cv2
import numpy as np
import tensorflow as tf
import pickle
import os
from utils import pil_to_bytes, bytes_to_base64
from PIL import Image

# Model & scaler paths
MODELS = {
    "Carbonate": ("Carbonate.h5", "Carbonate_scaler.save"),
    "Chlorine": ("Chlorine.h5", "Chlorine_scaler.save"),
    "Hardness": ("Hardness.h5", "Hardness_scaler.save"),
    "Nitrate": ("Nitrate.h5", "Nitrate_scaler.save"),
    "Nitrite": ("Nitrite.h5", "Nitrite_scaler.save"),
    "pH": ("pH.h5", "pH_scaler.save")  # missing scaler handled later
}

loaded_models = {}
loaded_scalers = {}

# Load models & scalers
for key, (model_path, scaler_path) in MODELS.items():
    loaded_models[key] = tf.keras.models.load_model(model_path)
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            loaded_scalers[key] = pickle.load(f)
    else:
        loaded_scalers[key] = None  # Missing scaler fallback

def process_image(path):
    """Process input image, return cropped patches + debug image as base64."""
    image = cv2.imread(path)
    orig_h, orig_w = image.shape[:2]
    resized = cv2.resize(image, (800, int(800 * orig_h / orig_w)))

    # HSV mask for pads
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 40, 40])
    upper_bound = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [(x, y, w, h) for (x, y, w, h) in (cv2.boundingRect(c) for c in contours) if w * h > 5000]
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    patches = []
    debug_img = resized.copy()
    for (x, y, w, h) in boxes:
        crop = resized[y:y+h, x:x+w]
        crop = cv2.resize(crop, (224, 224))
        patches.append(crop)
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Convert debug image to base64
    pil_img = Image.fromarray(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
    debug_b64 = bytes_to_base64(pil_to_bytes(pil_img))

    return patches, debug_b64

def predict_all(patches):
    """Predict all water quality parameters."""
    results = {}
    for param, model in loaded_models.items():
        scaler = loaded_scalers[param]
        param_preds = []

        for patch in patches:
            patch_norm = patch.astype("float32") / 255.0
            patch_norm = np.expand_dims(patch_norm, axis=0)
            pred_scaled = model.predict(patch_norm, verbose=0)[0][0]

            # Apply scaler if available, else manual scaling fallback
            if scaler is not None:
                try:
                    pred_val = float(scaler.inverse_transform([[pred_scaled]])[0][0])
                except Exception:
                    pred_val = float(pred_scaled)
            else:
                pred_val = float(pred_scaled * 14.0)  # Example: pH range fallback

            param_preds.append(round(pred_val, 3))
        results[param] = param_preds
    return results
