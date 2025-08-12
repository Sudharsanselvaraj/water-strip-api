import base64
import io
from PIL import Image

def pil_to_bytes(pil_img, fmt="JPEG", quality=85):
    bio = io.BytesIO()
    pil_img.save(bio, format=fmt, quality=quality)
    bio.seek(0)
    return bio.read()

def bytes_to_base64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")
