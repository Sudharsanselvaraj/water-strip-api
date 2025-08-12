from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import tensorflow as tf
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import io
import uvicorn
import datetime

BASE_DIR = Path(__file__).resolve().parent

# Load models and scalers
models = {
    "pH": (tf.keras.models.load_model(BASE_DIR / "models" / "pH.h5"), joblib.load(BASE_DIR / "models" / "pH_scaler.save")),
    "Nitrate": (tf.keras.models.load_model(BASE_DIR / "models" / "Nitrate.h5"), joblib.load(BASE_DIR / "models" / "Nitrate_scaler.save")),
    "Nitrite": (tf.keras.models.load_model(BASE_DIR / "models" / "Nitrite.h5"), joblib.load(BASE_DIR / "models" / "Nitrite_scaler.save")),
    "Chlorine": (tf.keras.models.load_model(BASE_DIR / "models" / "Chlorine.h5"), joblib.load(BASE_DIR / "models" / "Chlorine_scaler.save")),
    "Total Hardness": (tf.keras.models.load_model(BASE_DIR / "models" / "Hardness.h5"), joblib.load(BASE_DIR / "models" / "Hardness_scaler.save")),
    "Carbonate": (tf.keras.models.load_model(BASE_DIR / "models" / "Carbonate.h5"), joblib.load(BASE_DIR / "models" / "Carbonate_scaler.save")),
}

# Parameter safe ranges for status calculation
safe_ranges = {
    "pH": (6.5, 8.5),
    "Nitrate": (0, 10),
    "Nitrite": (0, 10),
    "Chlorine": (0, 1),
    "Total Hardness": (0, 100),
    "Carbonate": (0, 0.2),
}

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        results = {}
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        y_offset = 10

        for param, (model, scaler) in models.items():
            # Predict and inverse scale
            pred_scaled = model.predict(img_array)
            pred_value = scaler.inverse_transform(pred_scaled)[0][0]
            pred_value = float(round(pred_value, 2))

            # Determine status
            safe_min, safe_max = safe_ranges[param]
            status = "safe" if safe_min <= pred_value <= safe_max else "caution"

            results[param] = {
                "value": pred_value,
                "status": status,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %I:%M %p")
            }

            # Draw on debug image
            draw.text((10, y_offset), f"{param}: {pred_value} ({status})", fill="red", font=font)
            y_offset += 15

        # Save debug image to memory
        debug_img_bytes = io.BytesIO()
        img.save(debug_img_bytes, format="PNG")
        debug_img_bytes.seek(0)

        return JSONResponse(content={
            "parameters": results
        }, headers={"X-Debug-Image": "Use /debug to fetch the annotated image"})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug")
async def get_debug():
    return FileResponse("debug.png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
