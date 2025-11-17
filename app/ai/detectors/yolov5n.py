"""
YOLOv5n Detector
Ultralytics YOLOv5n 사용
"""

import numpy as np
from typing import List, Dict
from ultralytics import YOLO
from .base import BaseDetector


class YOLOv5nDetector(BaseDetector):
    """
    YOLOv5n 감지 모델
    """

    def __init__(self, model_path: str = "yolov5n.pt", confidence: float = 0.5):
        super().__init__(confidence)
        self.model = YOLO(model_path)
        print(f"[YOLOv5n] Loaded model: {model_path}")

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        이미지에서 사람 감지
        Args:
            image: OpenCV BGR 이미지 (H, W, C)
        """
        results = self.model(image, conf=self.confidence, verbose=False)

        detections: List[Dict] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                detections.append({
                    "box": box.tolist(),          # [x1, y1, x2, y2]
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                })

        # BaseDetector에 있는 사람 필터
        return self.filter_persons(detections)

    def warmup(self):
        """
        더미 이미지로 워밍업
        """
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy)
        print("[YOLOv5n] Warmup complete")
