"""
RT-DETR Detector
Ultralytics RT-DETR 사용
"""

import numpy as np
from typing import List, Dict

from ultralytics import RTDETR   
from .base import BaseDetector


class RTDetrDetector(BaseDetector):
    """
    RT-DETR 감지 모델
    """

    def __init__(self, model_path: str = "rtdetr-l.pt", confidence: float = 0.5):
        """
        Args:
            model_path: RT-DETR weight 경로 (예: "rtdetr-l.pt")
            confidence: score threshold
        """
        super().__init__(confidence)
        self.model = RTDETR(model_path)
        print(f"[RT-DETR] Loaded model: {model_path}")

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        이미지에서 사람 감지
        image: OpenCV BGR (H, W, C)
        """
        results = self.model(image, conf=self.confidence, verbose=False)

        detections: List[Dict] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()      # [N, 4]
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                detections.append(
                    {
                        "box": box.tolist(),       # [x1, y1, x2, y2]
                        "confidence": float(conf),
                        "class_id": int(cls_id),
                    }
                )

        return self.filter_persons(detections)

    def warmup(self):
        """
        더미 이미지로 워밍업
        """
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy)
        print("[RT-DETR] Warmup complete")
