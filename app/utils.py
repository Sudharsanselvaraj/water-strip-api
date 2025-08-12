import base64
import io
from PIL import Image
import numpy as np
import cv2


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_jpeg_bytes(img: np.ndarray, quality: int = 90) -> bytes:
    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes()


def jpeg_bytes_to_base64(b: bytes) -> str:
    return base64.b64encode(b).decode('utf-8')
