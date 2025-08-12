# app/process.py
import os
import cv2
import numpy as np
import joblib
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

# Adjust these to match model filenames
MODEL_DIR = os.path.join(os.getcwd(), "models")
# The parameters we expect (keys used in output)
PARAM_ORDER = ["Nitrate", "Nitrite", "Chlorine", "Hardness", "Carbonate", "pH"]

# attempt to load models & scalers (case-insensitive match)
def _find_file_for_param(param, ext_list):
    for f in os.listdir(MODEL_DIR):
        low = f.lower()
        for ext in ext_list:
            if low.endswith(ext) and param.lower() in low:
                return os.path.join(MODEL_DIR, f)
    return None

models = {}
scalers = {}
for p in PARAM_ORDER:
    mfile = _find_file_for_param(p, ['.h5', '.keras'])
    sfile = _find_file_for_param(p, ['.save', '.pkl', '.joblib'])
    if mfile:
        try:
            models[p] = tf.keras.models.load_model(mfile, compile=False)
            print(f"Loaded model for {p}: {os.path.basename(mfile)}")
        except Exception as e:
            print(f"Failed to load model {mfile} for {p}: {e}")
            models[p] = None
    else:
        print(f"No model file found for {p}")
        models[p] = None

    if sfile:
        try:
            scalers[p] = joblib.load(sfile)
            print(f"Loaded scaler for {p}: {os.path.basename(sfile)}")
        except Exception as e:
            print(f"Failed to load scaler {sfile} for {p}: {e}")
            scalers[p] = None
    else:
        print(f"No scaler file found for {p}")
        scalers[p] = None

# --- image detection & preprocessing utilities (basic, adjust as needed) ---
IMG_SIZE = (128, 128)

def preprocess_for_model_cv(img_bgr):
    img = cv2.resize(img_bgr, IMG_SIZE)
    arr = img.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def find_color_patches(img_bgr, expected_pads=len(PARAM_ORDER)):
    """
    Basic color-patch finder similar to earlier code.
    Returns list of boxes (x,y,w,h), left-to-right.
    If detection fails or count mismatch, return evenly split boxes.
    """
    h, w = img_bgr.shape[:2]
    blur = cv2.GaussianBlur(img_bgr, (5,5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    s_ch, v_ch = hsv[:,:,1], hsv[:,:,2]

    SAT_THRESHOLD = 25
    MIN_AREA_RATIO = 0.0005

    sat_mask = (s_ch > SAT_THRESHOLD).astype('uint8') * 255
    val_mask = (v_ch < 250).astype('uint8') * 255
    mask = cv2.bitwise_and(sat_mask, val_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x,y,ww,hh = cv2.boundingRect(cnt)
        area = ww*hh
        if area < (w*h)*MIN_AREA_RATIO:
            continue
        aspect = ww/float(hh) if hh>0 else 0
        if aspect < 0.3 or aspect > 3:
            continue
        boxes.append((x,y,ww,hh))
    boxes = sorted(boxes, key=lambda b: b[0])
    # fallback to equal vertical splits
    if len(boxes) != expected_pads:
        box_w = w // expected_pads
        boxes = [(i*box_w, 0, box_w, h) for i in range(expected_pads)]
    return boxes

def tight_crop(img, x,y,ww,hh, pad_h_ratio=0.5, pad_w_ratio=0.2):
    pad_h = int(hh * pad_h_ratio)
    pad_w = int(ww * pad_w_ratio)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img.shape[1], x + ww + pad_w)
    y2 = min(img.shape[0], y + hh + pad_h)
    return img[y1:y2, x1:x2]

# Main function used by app.main
def predict_from_pil_image(pil_img):
    """
    Input: PIL.Image
    Output: (results_dict, debug_pil_image)
    results_dict: { param: { value, unit, safety } ... }
    """
    # Convert to CV2 BGR array
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    boxes = find_color_patches(img_bgr, expected_pads=len(PARAM_ORDER))

    results = {}
    annotated = img_np.copy()
    # draw with PIL for simpler text rendering
    annotated_pil = pil_img.copy()
    draw = ImageDraw.Draw(annotated_pil)

    # Use a simple default font via PIL
    try:
        font = ImageFont.load_default()
    except:
        font = None

    for i, p in enumerate(PARAM_ORDER):
        x,y,ww,hh = boxes[i]
        crop = tight_crop(img_bgr, x,y,ww,hh)
        # If no model, set None
        model = models.get(p)
        scaler = scalers.get(p)
        pred_val = None
        if model is not None:
            inp = preprocess_for_model_cv(crop)
            # model output may be scaled; we assume scaler was fit on training target
            try:
                pred_scaled = float(model.predict(inp, verbose=0)[0][0])
                if scaler is not None:
                    try:
                        pred_val = float(scaler.inverse_transform([[pred_scaled]])[0][0])
                    except Exception:
                        pred_val = float(pred_scaled)
                else:
                    pred_val = float(pred_scaled)
            except Exception as e:
                pred_val = None
        # unit and simple safety
        unit = "ppm" if p.lower() != "pH" else "pH"
        safety = classify_status(p, pred_val) if pred_val is not None else "unknown"

        results[p if p!="Hardness" else "Total Hardness"] = {
            "value": round(pred_val, 3) if pred_val is not None else None,
            "unit": unit,
            "safety": safety
        }

        # annotate rectangle & text
        cv2.rectangle(annotated, (x,y), (x+ww, y+hh), (0,255,0), 2)
        text = f"{p}: {results[p if p!='Hardness' else 'Total Hardness']['value']}"
        # Put text on PIL image
        draw.text((x, max(0,y-12)), text, fill="red", font=font)

    # Convert annotated_pil back to PIL.Image (already annotated)
    return results, annotated_pil

# local copy of classify_status to reduce imports for simplicity
from PIL import ImageFont, ImageDraw
def classify_status(param, value):
    """If value is None, returns 'unknown'"""
    if value is None:
        return "unknown"
    try:
        v = float(value)
    except Exception:
        return "unknown"

    # thresholds (customize as needed)
    if param.lower() == "ph":
        if 6.5 <= v <= 8.5:
            return "safe"
        return "caution"
    if param.lower() == "hardness":
        if v < 150: return "safe"
        if v < 300: return "caution"
        return "danger"
    # generic for other ppm params
    if v <= 1: return "safe"
    if v <= 5: return "caution"
    return "danger"
