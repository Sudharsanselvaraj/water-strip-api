import cv2
import numpy as np
import tensorflow as tf
import pickle

# Load all models & scalers from your current directory
MODELS = {
    "Carbonate": ("Carbonate.h5", "Carbonate_scaler.save"),
    "Chlorine": ("Chlorine.h5", "Chlorine_scaler.save"),
    "Hardness": ("Hardness.h5", "Hardness_scaler.save"),
    "Nitrate": ("Nitrate.h5", "Nitrate_scaler.save"),
    "Nitrite": ("Nitrite.h5", "Nitrite_scaler.save"),
    "pH": ("pH.h5", "pH_scaler.save")
}

loaded_models = {}
loaded_scalers = {}

for key, (model_path, scaler_path) in MODELS.items():
    loaded_models[key] = tf.keras.models.load_model(model_path)
    with open(scaler_path, "rb") as f:
        loaded_scalers[key] = pickle.load(f)

def find_color_patches(image):
    orig_h, orig_w = image.shape[:2]
    image_resized = cv2.resize(image, (800, int(800 * orig_h / orig_w)))

    hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 40, 40])
    upper_bound = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    patches = [(x, y, w, h) for (x, y, w, h) in (cv2.boundingRect(c) for c in contours) if w * h > 5000]
    patches = sorted(patches, key=lambda b: (b[1], b[0]))
    return patches, image_resized

def process_image(path):
    image = cv2.imread(path)
    boxes, resized_img = find_color_patches(image)
    patch_images = []
    for (x, y, w, h) in boxes:
        crop = resized_img[y:y+h, x:x+w]
        crop = cv2.resize(crop, (224, 224))
        patch_images.append(crop)
    return patch_images

def predict_all(patches):
    results = {}
    for param, model in loaded_models.items():
        scaler = loaded_scalers[param]
        param_preds = []
        for patch in patches:
            patch_norm = patch.astype("float32") / 255.0
            patch_norm = np.expand_dims(patch_norm, axis=0)
            pred_scaled = model.predict(patch_norm, verbose=0)[0][0]
            try:
                pred_val = float(scaler.inverse_transform([[pred_scaled]])[0][0])
            except Exception:
                pred_val = float(pred_scaled)
            param_preds.append(pred_val)
        results[param] = param_preds
    return results
