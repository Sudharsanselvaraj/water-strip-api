# app/main.py
import math

def sanitize_for_json(obj):
    """Recursively replace NaN/inf with safe numbers or None."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return 1e9 if obj > 0 else -1e9
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj

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
        results, debug_img = predict_from_pil_image(pil)

        # Sanitize output
        results = sanitize_for_json(results)

        filename = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        debug_path = os.path.join(DEBUG_DIR, filename)
        debug_img.save(debug_path, format="JPEG", quality=85)

        return JSONResponse(content={
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
            "predictions": results,
            "debug_image_url": f"/debug/{filename}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
