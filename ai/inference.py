from ultralytics import YOLO

model = YOLO("yolo11n.pt")


# image should be a path, PIL image, OpenCV image, numpy array, or torch tensor
#
# path: str
#     e.g. image = "image.jpg"
# PIL image: PIL.Image with HWC + RGB
#     e.g. from PIL import Image; image = Image.open("image.jpg")
# OpenCV image: np.ndarray with HWC + BGR (0~255)
#     e.g. import cv2; image = cv2.imread("image.jpg")
# numpy array: np.ndarray with HWC + BGR (0~255)
#     e.g. import numpy as np; image = np.zeros((640, 1280, 3))
# torch tensor: torch.Tensor with BCHW + RGB (0~1)
#     e.g. import torch; image = torch.zeros(16, 3, 320, 640)
def inference(image):
    result = model.track(image, persist=True)[0]

    human_results = []

    if result.boxes and result.boxes.is_track:
        boxes = result.boxes.xyxy.int().cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()
        track_ids = result.boxes.id.int().cpu().tolist()

        # Filter only person class
        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            if class_id == 0:  # person class
                human_results.append({"box": box, "track_id": track_id})

    return human_results
