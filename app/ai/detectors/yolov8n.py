"""
YOLOv8n Detector (YOLO11n보다 빠름)
Ultralytics YOLOv8n 사용
"""

import numpy as np
from ultralytics import YOLO
from typing import List, Dict
from .base import BaseDetector


class YOLOv8nDetector(BaseDetector):
    """
    YOLOv8n 감지 모델
    - YOLO11n보다 약간 더 빠름
    - 정확도는 비슷
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5):
        super().__init__(confidence)
        self.model = YOLO(model_path)
        print(f"[YOLOv8n] Loaded model: {model_path}")

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        이미지에서 사람 감지
        """
        results = self.model(image, conf=self.confidence, verbose=False)
        
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                detections.append({
                    "box": box.tolist(),
                    "confidence": float(conf),
                    "class_id": int(cls_id)
                })
        
        return self.filter_persons(detections)

    def warmup(self):
        """
        더미 이미지로 워밍업
        """
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy)
        print("[YOLOv8n] Warmup complete")
