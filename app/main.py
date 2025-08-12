import os
import io
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytz
from app.process import predict_from_pil_image

DEBUG_DIR = os.path.join(os.getcwd(), "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

app = FastAPI(title="Water Strip Analyzer")

# CORS middleware — allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Allow all origins - everyone can access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/debug", StaticFiles(directory=DEBUG_DIR), name="debug")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        contents = await file.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {e}")

    try:
        results, debug_img, overall_quality, quality_description = predict_from_pil_image(pil)

        filename = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        debug_path = os.path.join(DEBUG_DIR, filename)
        debug_img.save(debug_path, format="JPEG", quality=85)

        base_url = "https://water-strip-api.onrender.com"
        debug_url = f"{base_url}/debug/{filename}"

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
