# app/process.py
import os
import cv2
import numpy as np
import joblib
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import math
import traceback

MODEL_DIR = os.path.join(os.getcwd(), "models")
PARAM_ORDER = ["Nitrate", "Nitrite", "Chlorine", "Hardness", "Carbonate", "pH"]

# Helpers to detect scaler type
def is_minmax_scaler(scaler):
    return hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_")

def scaler_range(scaler):
    try:
        return (float(getattr(scaler, "data_min_[0]", math.nan)),
                float(getattr(scaler, "data_max_[0]", math.nan)))
    except Exception:
        return (math.nan, math.nan)

# load models & scalers
models = {}
scalers = {}
for p in PARAM_ORDER:
    models[p] = None
    scalers[p] = None
    # find model
    for f in os.listdir(MODEL_DIR):
        if f.lower().endswith((".h5", ".keras")) and p.lower() in f.lower():
            try:
                models[p] = tf.keras.models.load_model(os.path.join(MODEL_DIR, f), compile=False)
                print(f"[process] Loaded model for {p}: {f}")
            except Exception as e:
                print(f"[process] Failed to load model {f} for {p}: {e}")
            break
    # find scaler
    for f in os.listdir(MODEL_DIR):
        if f.lower().endswith((".save", ".pkl", ".joblib")) and p.lower() in f.lower():
            try:
                scalers[p] = joblib.load(os.path.join(MODEL_DIR, f))
                print(f"[process] Loaded scaler for {p}: {f}")
            except Exception as e:
                print(f"[process] Failed to load scaler {f} for {p}: {e}")
            break

# image / model preprocessing
IMG_SIZE = (128, 128)

def preprocess_for_model_cv(img_bgr):
    # expect BGR OpenCV image; resize and scale to 0..1
    img = cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    arr = img.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

# adaptive patch detection that falls back to equal split
def find_color_patches(img_bgr, expected_pads=len(PARAM_ORDER)):
    h, w = img_bgr.shape[:2]
    blur = cv2.GaussianBlur(img_bgr, (5,5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    # make threshold adaptive depending on median saturation
    median_s = int(np.median(s))
    sat_thresh = max(12, int(median_s * 0.5))  # throttled for pale strips

    sat_mask = (s >= sat_thresh).astype("uint8") * 255
    val_mask = (v < 250).astype("uint8") * 255

    # LAB L channel to capture non-white texture
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    l = lab[:,:,0]
    nonwhite = (l < 245).astype("uint8") * 255

    mask = cv2.bitwise_or(cv2.bitwise_and(sat_mask, val_mask), nonwhite)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    MIN_AREA_RATIO = 0.0002
    for cnt in contours:
        x,y,ww,hh = cv2.boundingRect(cnt)
        if ww*hh < (w*h)*MIN_AREA_RATIO:
            continue
        aspect = ww/float(hh) if hh>0 else 0
        if aspect < 0.3 or aspect > 3.5:
            continue
        boxes.append((x,y,ww,hh))

    boxes = sorted(boxes, key=lambda b: b[0])

    # if detection count not match, fallback to even vertical split
    if len(boxes) != expected_pads:
        box_w = w // expected_pads
        boxes = [(i*box_w, 0, box_w, h) for i in range(expected_pads)]
        # return those boxes
    return boxes

def tight_crop(img, x, y, w, h, pad_h_ratio=0.5, pad_w_ratio=0.2):
    pad_h = int(h * pad_h_ratio)
    pad_w = int(w * pad_w_ratio)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img.shape[1], x + w + pad_w)
    y2 = min(img.shape[0], y + h + pad_h)
    return img[y1:y2, x1:x2]

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
        if v < 150: return "safe"
        if v < 300: return "caution"
        return "danger"
    if v <= 1: return "safe"
    if v <= 5: return "caution"
    return "danger"

def predict_from_pil_image(pil_img):
    """
    Input: PIL image (RGB)
    Output: (results_dict, annotated_pil_image)
    results: { param_or_Total Hardness: {value, unit, safety, debug: {raw_pred, used_scaler}} }
    """
    try:
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise RuntimeError(f"Cannot convert PIL to BGR: {e}")

    boxes = find_color_patches(img_bgr, expected_pads=len(PARAM_ORDER))

    annotated_pil = pil_img.copy()
    draw = ImageDraw.Draw(annotated_pil)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    results = {}
    for i, p in enumerate(PARAM_ORDER):
        x,y,ww,hh = boxes[i]
        crop = tight_crop(img_bgr, x, y, ww, hh)
        model = models.get(p)
        scaler = scalers.get(p)
        raw_pred = None
        final_val = None
        scaler_used = None
        try:
            if model is not None:
                inp = preprocess_for_model_cv(crop)
                raw_pred = float(model.predict(inp, verbose=0)[0][0])
                # Decide whether to inverse_transform
                if scaler is not None and hasattr(scaler, "inverse_transform"):
                    # If scaler looks like MinMax (has data_min_), use inverse_transform
                    if is_minmax_scaler(scaler):
                        try:
                            final_val = float(scaler.inverse_transform([[raw_pred]])[0][0])
                            scaler_used = "minmax_inverse"
                        except Exception as e:
                            # fallback: maybe model already outputs original scale
                            final_val = float(raw_pred)
                            scaler_used = f"inverse_failed:{e}"
                    else:
                        # scaler exists but not MinMax (maybe StandardScaler) -> try inverse_transform but safely
                        try:
                            final_val = float(scaler.inverse_transform([[raw_pred]])[0][0])
                            scaler_used = "other_inverse"
                        except Exception as e:
                            final_val = float(raw_pred)
                            scaler_used = f"no_inverse:{e}"
                else:
                    final_val = float(raw_pred)
                    scaler_used = None
            else:
                raw_pred = None
                final_val = None
        except Exception as e:
            raw_pred = None
            final_val = None
            print(f"[predict] Exception for {p}: {e}\n{traceback.format_exc()}")

        key = p if p != "Hardness" else "Total Hardness"
        results[key] = {
            "value": round(final_val, 3) if final_val is not None else None,
            "unit": "pH" if p.lower() == "ph" else "ppm",
            "safety": classify_status(p, final_val),
            "debug": {
                "raw_pred": None if raw_pred is None else round(float(raw_pred), 6),
                "scaler_used": scaler_used,
                "scaler_range": scaler_range(scaler) if scaler is not None else None,
                "model_loaded": models.get(p) is not None
            }
        }

        # annotate
        draw.rectangle([(x, y), (x + ww, y + hh)], outline="lime", width=2)
        label = f"{p}: {results[key]['value']}"
        draw.text((x, max(0, y - 12)), label, fill="red", font=font)

    return results, annotated_pil
