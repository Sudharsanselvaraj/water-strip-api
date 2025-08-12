import os
import re
import cv2
import numpy as np
from typing import Dict, Tuple
import joblib
import tensorflow as tf

MODEL_DIR = os.environ.get('MODEL_DIR', 'models')
PARAM_ORDER = ["Nitrate", "Nitrite", "Chlorine", "Hardness", "Carbonate", "pH"]
IMG_SIZE = (128, 128)
SAT_THRESHOLD = 25
MIN_AREA_RATIO = 0.0005
PAD_H = 0.8
PAD_W = 0.8


# Load models and scalers once
models = {}
scalers = {}


def find_model_file(keyword: str):
    for f in os.listdir(MODEL_DIR):
        if f.lower().endswith(('.h5', '.keras')) and keyword.lower() in f.lower():
            return os.path.join(MODEL_DIR, f)
    return None


def find_scaler_file(keyword: str):
    for f in os.listdir(MODEL_DIR):
        if f.lower().endswith('.save') and keyword.lower() in f.lower():
            return os.path.join(MODEL_DIR, f)
    return None


for p in PARAM_ORDER:
    mp = find_model_file(p)
    sp = find_scaler_file(p)
    if mp:
        try:
            models[p] = tf.keras.models.load_model(mp, compile=False)
        except Exception as e:
            models[p] = None
    else:
        models[p] = None

    if sp:
        try:
            scalers[p] = joblib.load(sp)
        except Exception:
            scalers[p] = None
    else:
        scalers[p] = None


# --- Image processing helpers (basic color patch detection) ---

def find_color_patches(img_bgr: np.ndarray, expected_pads: int = 6) -> list:
    h, w = img_bgr.shape[:2]
    blur = cv2.GaussianBlur(img_bgr, (5,5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    s_ch, v_ch = hsv[:,:,1], hsv[:,:,2]

    sat_mask = (s_ch > SAT_THRESHOLD).astype(np.uint8)*255
    val_mask = (v_ch < 250).astype(np.uint8)*255
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
        aspect = ww/float(hh)
        if aspect < 0.3 or aspect > 3:
            continue
        boxes.append((x,y,ww,hh))

    boxes = sorted(boxes, key=lambda b: b[0])
    if len(boxes) != expected_pads:
        # fallback even split
        box_w = w // expected_pads
        boxes = [(i*box_w, 0, box_w, h) for i in range(expected_pads)]
    return boxes


def tight_crop(img: np.ndarray, x:int,y:int,ww:int,hh:int, pad_h_ratio=PAD_H, pad_w_ratio=PAD_W) -> np.ndarray:
    pad_h = int(hh*pad_h_ratio)
    pad_w = int(ww*pad_w_ratio)
    x1 = max(0, x-pad_w)
    y1 = max(0, y-pad_h)
    x2 = min(img.shape[1], x+ww+pad_w)
    y2 = min(img.shape[0], y+hh+pad_h)
    return img[y1:y2, x1:x2]


def preprocess_for_model(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, IMG_SIZE)
    arr = img.astype(np.float32)/255.0
    return np.expand_dims(arr, axis=0)


# --- Prediction runner ---

def predict_from_strip(img_bgr: np.ndarray) -> Tuple[dict, np.ndarray]:
    boxes = find_color_patches(img_bgr, expected_pads=len(PARAM_ORDER))
    results = {}
    vis = img_bgr.copy()

    for idx, (param, box) in enumerate(zip(PARAM_ORDER, boxes)):
        x,y,ww,hh = box
        crop = tight_crop(img_bgr, x,y,ww,hh)
        inp = preprocess_for_model(crop)
        model = models.get(param)
        scaler = scalers.get(param)
        if model is not None:
            pred_scaled = float(model.predict(inp, verbose=0)[0][0])
            if scaler is not None:
                try:
                    pred = float(scaler.inverse_transform([[pred_scaled]])[0][0])
                except Exception:
                    pred = float(pred_scaled)
            else:
                pred = float(pred_scaled)
            results[param] = round(pred,3)
        else:
            results[param] = None

        # annotate
        cv2.rectangle(vis, (x,y),(x+ww, y+hh),(0,255,0),2)
        label = f"{param}:{results[param]}"
        cv2.putText(vis, label, (x, max(12,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)

    return results, vis
