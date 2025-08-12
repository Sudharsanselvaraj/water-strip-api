import os
import io
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Add pytz for timezone support
import pytz

# Import from app package
from app.process import predict_from_pil_image

# Config
DEBUG_DIR = os.path.join(os.getcwd(), "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

app = FastAPI(title="Water Strip Analyzer")

# Serve debug images at /debug/<filename>
app.mount("/debug", StaticFiles(directory=DEBUG_DIR), name="debug")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Analyze uploaded image and return predictions + debug image URL + overall quality."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        contents = await file.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {e}")

    try:
        # Run prediction - now returns 4 values including overall quality info
        results, debug_img, overall_quality, quality_description = predict_from_pil_image(pil)

        # Save debug image to disk
        filename = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        debug_path = os.path.join(DEBUG_DIR, filename)
        debug_img.save(debug_path, format="JPEG", quality=85)

        # Return full absolute URL
        base_url = "https://water-strip-api.onrender.com"
        debug_url = f"{base_url}/debug/{filename}"

        # Use timezone-aware timestamp (example: Asia/Kolkata)
        tz = pytz.timezone("Asia/Kolkata")
        now = datetime.now(tz)
        timestamp_str = now.strftime("%Y-%m-%d %I:%M %p %Z")

        return JSONResponse(content={
            "status": "success",
            "timestamp": timestamp_str,
            "predictions": results,
            "overall_quality": overall_quality,
            "quality_description": quality_description,
            "debug_image_url": debug_url
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
