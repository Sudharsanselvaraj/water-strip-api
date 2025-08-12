from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
from app.process import process_image, predict_all
from app.utils import some_function


app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        patches, debug_img_b64 = process_image(temp_path)
        results = predict_all(patches)

        return JSONResponse(content={
            "predictions": results,
            "debug_image": debug_img_b64
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
