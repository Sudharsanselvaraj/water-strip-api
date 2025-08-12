import io
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from .process import predict_from_strip
from .utils import pil_to_cv2, cv2_to_jpeg_bytes, jpeg_bytes_to_base64

app = FastAPI(title='Water Strip Analyzer')
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post('/analyze')
async def analyze(file: UploadFile = File(...)):
    if file.content_type.split('/')[0] != 'image':
        raise HTTPException(status_code=400, detail='file must be an image')

    contents = await file.read()
    pil = Image.open(io.BytesIO(contents)).convert('RGB')
    img = pil_to_cv2(pil)

    results, vis = predict_from_strip(img)

    jpeg = cv2_to_jpeg_bytes(vis)
    debug_b64 = jpeg_bytes_to_base64(jpeg)

    response = {
        'status': 'success',
        'predictions': {},
        'debug_image_base64': debug_b64
    }

    # add units & simple safety logic
    for k,v in results.items():
        unit = 'ppm' if k.lower() != 'pH' else 'pH'
        safe = 'unknown'
        if v is None:
            safe = 'unknown'
        else:
            if k == 'pH':
                if 6.5 <= v <= 8.5:
                    safe = 'safe'
                else:
                    safe = 'caution'
            elif k == 'Hardness':
                if v < 150:
                    safe = 'safe'
                elif v < 300:
                    safe = 'caution'
                else:
                    safe = 'danger'
            else:
                # generic thresholds, please customize
                if v <= 1:
                    safe = 'safe'
                elif v <= 5:
                    safe = 'caution'
                else:
                    safe = 'danger'

        response['predictions'][k] = {
            'value': v,
            'unit': unit,
            'safety': safe
        }

    return response
