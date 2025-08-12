import os
import cv2
import numpy as np
import joblib
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from app.utils import pil_to_bytes, bytes_to_base64

# ------------------------
# Model configuration
# ------------------------
MODEL_DIR = os.path.join(os.getcwd(), "models")
PARAM_ORDER = ["Nitrate", "Nitrite", "Chlorine", "Hardness", "Carbonate", "pH"]
IMG_SIZE = (128, 128)

# ------------------------
# Helper: Find model file
# ------------------------
def _find_file_for_param(param, ext_list):
    for f in os.listdir(MODEL_DIR):
        low = f.lower()
        if any(low.endswith(ext) and param.lower() in low for ext in ext_list):
            return os.path.join(MODEL_DIR, f)
    return None

# ------------------------
# Load models
# ------------------------
models = {}
for p in PARAM_ORDER:
    mfile = _find_file_for_param(p, [".h5", ".keras"])
    if mfile:
        try:
            models[p] = tf.keras.models.load_model(mfile, compile=False)
            print(f"✅ Loaded model for {p}: {os.path.basename(mfile)}")
        except Exception as e:
            print(f"❌ Failed to load model for {p}: {e}")
            models[p] = None
    else:
        print(f"⚠ No model file found for {p}")
        models[p] = None

# ------------------------
# Preprocessing
# ------------------------
def preprocess_for_model_cv(img_bgr):
    img = cv2.resize(img_bgr, IMG_SIZE)
    arr = img.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

# ------------------------
# Equal-split pad detection (consistent crops)
# ------------------------
def find_color_patches_equal_split(img_bgr, expected_pads=len(PARAM_ORDER)):
    """Split image evenly into vertical sections for pad cropping."""
    h, w = img_bgr.shape[:2]
    box_w = w // expected_pads
    return [(i * box_w, 0, box_w, h) for i in range(expected_pads)]

# ------------------------
# Crop with padding
# ------------------------
def tight_crop(img, x, y, w, h, pad_h_ratio=0.2, pad_w_ratio=0.2):
    pad_h = int(h * pad_h_ratio)
    pad_w = int(w * pad_w_ratio)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img.shape[1], x + w + pad_w)
    y2 = min(img.shape[0], y + h + pad_h)
    return img[y1:y2, x1:x2]

# ------------------------
# Status classification
# ------------------------
def classify_status(param, value):
    if value is None:
        return "unknown"
    try:
        v = float(value)
    except Exception:
        return "unknown"

    if param.lower() == "ph":
        return "safe" if 6.5 <= v <= 8.5 else "caution"
    if param.lower() == "hardness":
        return "safe" if v < 150 else "caution" if v < 300 else "danger"
    if v <= 1:
        return "safe"
    if v <= 5:
        return "caution"
    return "danger"

# ------------------------
# Prediction pipeline
# ------------------------
def predict_from_pil_image(pil_img):
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Always use equal split for consistent crops
    boxes = find_color_patches_equal_split(img_bgr, expected_pads=len(PARAM_ORDER))

    annotated_pil = pil_img.copy()
    draw = ImageDraw.Draw(annotated_pil)
    font = ImageFont.load_default()

    results = {}
    for i, param in enumerate(PARAM_ORDER):
        x, y, ww, hh = boxes[i]
        crop = tight_crop(img_bgr, x, y, ww, hh)

        model = models.get(param)
        pred_val = None

        if model is not None:
            try:
                inp = preprocess_for_model_cv(crop)
                pred_val = float(model.predict(inp, verbose=0)[0][0])  # Direct output, no scaler
            except Exception as e:
                print(f"Prediction failed for {param}: {e}")
                pred_val = None

        key = param if param != "Hardness" else "Total Hardness"
        results[key] = {
            "value": round(pred_val, 3) if pred_val is not None else None,
            "unit": "pH" if param.lower() == "ph" else "ppm",
            "safety": classify_status(param, pred_val)
        }

        # Draw bounding boxes
        draw.rectangle([(x, y), (x + ww, y + hh)], outline="lime", width=2)
        label = f"{param}: {results[key]['value']}"
        draw.text((x, max(0, y - 12)), label, fill="red", font=font)

    return results, annotated_pil
