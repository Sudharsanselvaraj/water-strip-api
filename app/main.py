from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
from process import process_image, predict_all

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        patches = process_image(temp_path)
        results = predict_all(patches)  # Calls your existing prediction logic for all parameters

        return JSONResponse(content={"predictions": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
