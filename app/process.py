import numpy as np
from PIL import Image, ImageDraw

def process_strip_image(img: Image.Image, debug_img_path: str):
    """
    Process uploaded strip image to extract features for ML models.
    Save debug image with markings.
    """
    # Convert to RGB & resize (example: 224x224 for CNNs)
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0

    # Example: Flatten features (depends on your model input)
    features = arr.flatten()

    # Save debug image (with a red box example)
    debug_img = img.copy()
    draw = ImageDraw.Draw(debug_img)
    draw.rectangle([(10, 10), (50, 50)], outline="red", width=3)
    debug_img.save(debug_img_path)

    return features
