import cv2
import base64
import numpy as np


def from_base64(img_b64, format):
    img_data = base64.b64decode(img_b64)
    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


def to_base64(image, format):
    _, buffer = cv2.imencode(f".{format}", image)
    img_b64 = base64.b64encode(buffer).decode("utf-8")
    return img_b64
