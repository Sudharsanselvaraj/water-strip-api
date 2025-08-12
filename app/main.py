import os
import io
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image

from process import process_strip_image
from utils import classify_status

# Create app
app = FastAPI()

# Paths
MODEL_DIR = "models"
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Mount static debug images
app.mount("/debug", StaticFiles(directory=DEBUG_DIR), name="debug")

# Parameter metadata
PARAM_INFO = {
    "pH": {
        "description": "Measures acidity/alkalinity. Critical for drinking water safety and taste.",
        "health": "Extreme pH levels can cause skin/eye irritation and affect mineral absorption."
    },
    "Nitrate": {
        "description": "Measures nitrate concentration in water.",
        "health": "High nitrate levels can cause health issues, especially in infants."
    },
    "Nitrite": {
        "description": "Measures nitrite concentration in water.",
        "health": "Nitrite can interfere with oxygen transport in the blood."
    },
    "Chlorine": {
        "description": "Measures residual chlorine in water.",
        "health": "Too much chlorine can cause irritation; too little may allow bacterial growth."
    },
    "Total Hardness": {
        "description": "Measures calcium and magnesium content.",
        "health": "High hardness can cause scaling and affect soap efficiency."
    },
    "Carbonate": {
        "description": "Measures carbonate concentration.",
        "health": "Carbonates affect pH balance and alkalinity."
    }
}

# Load models and scalers
models = {}
scalers = {}

for param in ["pH", "Nitrate", "Nitrite", "Chlorine", "Hardness", "Carbonate"]:
    h5_path = os.path.join(MODEL_DIR, f"{param.replace(' ', '')}.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{param.replace(' ', '')}_scaler.save")

    if os.path.exists(h5_path) and os.path.exists(scaler_path):
        models[param] = tf.keras.models.load_model(h5_path)
        scalers[param] = joblib.load(scaler_path)
    else:
        print(f"⚠ Missing model or scaler for {param}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded image temporarily
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Generate debug image path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_img_path = os.path.join(DEBUG_DIR, f"debug_{timestamp}.jpg")
        
        # Process image (extract features for prediction)
        features = process_strip_image(img, debug_img_path)

        # Prepare output
        results = {}

        for param in models:
            scaler = scalers[param]
            model = models[param]

            scaled_features = scaler.transform([features])
            prediction = model.predict(scaled_features)[0][0]

            status = classify_status(param, prediction)

            results[param] = {
                "value": round(float(prediction), 2),
                "status": status,
                "description": PARAM_INFO[param]["description"],
                "health_effects": PARAM_INFO[param]["health"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p")
            }

        return JSONResponse(content={
            "results": results,
            "debug_image": f"/debug/{os.path.basename(debug_img_path)}"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
