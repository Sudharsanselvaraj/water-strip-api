import os
import io
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .process import predict_from_pil_image

# Config
DEBUG_DIR = os.path.join(os.getcwd(), "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

app = FastAPI(title="Water Strip Analyzer")

# Serve debug images at /debug/<filename>
app.mount("/debug", StaticFiles(directory=DEBUG_DIR), name="debug")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Analyze uploaded image and return predictions + debug image URL."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        contents = await file.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {e}")

    try:
        # Run prediction
        results, debug_img = predict_from_pil_image(pil)

        # Save debug image to disk
        filename = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        debug_path = os.path.join(DEBUG_DIR, filename)
        debug_img.save(debug_path, format="JPEG", quality=85)

        # Build response
        return JSONResponse(content={
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
            "predictions": results,
            "debug_image_url": f"/debug/{filename}"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
