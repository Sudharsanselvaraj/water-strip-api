import os
import cv2
import numpy as np
import joblib
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

# Model config
MODEL_DIR = os.path.join(os.getcwd(), "models")
PARAM_ORDER = ["Nitrate", "Nitrite", "Chlorine", "Hardness", "Carbonate", "pH"]

def _find_file_for_param(param, ext_list):
    """Find a file in MODEL_DIR matching parameter name and extension."""
    for f in os.listdir(MODEL_DIR):
        if any(f.lower().endswith(ext) and param.lower() in f.lower() for ext in ext_list):
            return os.path.join(MODEL_DIR, f)
    return None

# Load models and scalers
models, scalers = {}, {}
for p in PARAM_ORDER:
    # Load model
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

    # Load scaler
    sfile = _find_file_for_param(p, [".save", ".pkl", ".joblib"])
    if sfile:
        try:
            scalers[p] = joblib.load(sfile)
            print(f"✅ Loaded scaler for {p}: {os.path.basename(sfile)}")
        except Exception as e:
            print(f"❌ Failed to load scaler for {p}: {e}")
            scalers[p] = None
    else:
        scalers[p] = None

# Image processing settings
IMG_SIZE = (128, 128)

def preprocess_for_model_cv(img_bgr):
    img = cv2.resize(img_bgr, IMG_SIZE)
    arr = img.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def find_color_patches(img_bgr, expected_pads=len(PARAM_ORDER)):
    """Detect color patches or fall back to equal splits."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    s_ch, v_ch = hsv[:, :, 1], hsv[:, :, 2]

    mask = cv2.inRange(s_ch, 25, 255) & cv2.inRange(v_ch, 0, 250)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (5, 5), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (5, 5), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [(x, y, w_, h_) for x, y, w_, h_ in (cv2.boundingRect(c) for c in contours)
             if w_ * h_ > (w * h) * 0.0005]

    boxes = sorted(boxes, key=lambda b: b[0])
    if len(boxes) != expected_pads:  # fallback
        box_w = w // expected_pads
        boxes = [(i * box_w, 0, box_w, h) for i in range(expected_pads)]
    return boxes

def tight_crop(img, x, y, w, h, pad_h_ratio=0.5, pad_w_ratio=0.2):
    pad_h, pad_w = int(h * pad_h_ratio), int(w * pad_w_ratio)
    return img[max(0, y - pad_h):min(img.shape[0], y + h + pad_h),
               max(0, x - pad_w):min(img.shape[1], x + w + pad_w)]

def classify_status(param, value):
    """Classify safety level."""
    if value is None:
        return "unknown"
    try:
        v = float(value)
    except ValueError:
        return "unknown"

    if param.lower() == "ph":
        return "safe" if 6.5 <= v <= 8.5 else "caution"
    if param.lower() == "hardness":
        return "safe" if v < 150 else "caution" if v < 300 else "danger"
    return "safe" if v <= 1 else "caution" if v <= 5 else "danger"

def predict_from_pil_image(pil_img):
    """Run full pipeline: detect patches, predict, annotate image."""
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    boxes = find_color_patches(img_bgr)

    annotated_pil = pil_img.copy()
    draw = ImageDraw.Draw(annotated_pil)
    font = ImageFont.load_default()

    results = {}
    for i, param in enumerate(PARAM_ORDER):
        x, y, w_, h_ = boxes[i]
        crop = tight_crop(img_bgr, x, y, w_, h_)
        model, scaler = models.get(param), scalers.get(param)

        pred_val = None
        if model is not None:
            try:
                pred_scaled = float(model.predict(preprocess_for_model_cv(crop), verbose=0)[0][0])
                pred_val = float(scaler.inverse_transform([[pred_scaled]])[0][0]) if scaler else pred_scaled
            except Exception as e:
                print(f"Prediction failed for {param}: {e}")

        results[param if param != "Hardness" else "Total Hardness"] = {
            "value": round(pred_val, 3) if pred_val is not None else None,
            "unit": "pH" if param.lower() == "ph" else "ppm",
            "safety": classify_status(param, pred_val)
        }

        draw.text((x, max(0, y - 12)), f"{param}: {results[param if param != 'Hardness' else 'Total Hardness']['value']}", fill="red", font=font)

    return results, annotated_pil
