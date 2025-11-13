"""
MobileNet-SSD Detector (가장 빠름)
OpenCV DNN 모듈 사용
"""

import numpy as np
import cv2
from typing import List, Dict
from .base import BaseDetector


class MobileNetSSDDetector(BaseDetector):
    """
    MobileNet-SSD 감지 모델
    - 가장 빠른 속도
    - 정확도는 YOLO보다 낮음
    - CPU 최적화
    """

    def __init__(self, confidence: float = 0.5):
        super().__init__(confidence)
        
        # MobileNet-SSD 모델 로드 (OpenCV DNN)
        # COCO 클래스: person은 15번
        self.net = cv2.dnn.readNetFromCaffe(
            "models/MobileNetSSD_deploy.prototxt",
            "models/MobileNetSSD_deploy.caffemodel"
        )
        
        # CPU 백엔드 설정
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        print("[MobileNet-SSD] Loaded model")

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        이미지에서 사람 감지
        """
        h, w = image.shape[:2]
        
        # Blob 생성 (300x300 입력)
        blob = cv2.dnn.blobFromImage(
            image, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=True
        )
        
        # 추론
        self.net.setInput(blob)
        detections_raw = self.net.forward()
        
        detections = []
        for i in range(detections_raw.shape[2]):
            confidence = detections_raw[0, 0, i, 2]
            class_id = int(detections_raw[0, 0, i, 1])
            
            # 사람(class 15) + confidence 필터
            if class_id == 15 and confidence >= self.confidence:
                # 좌표 변환
                box = detections_raw[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                detections.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(confidence),
                    "class_id": 0  # 통일: person = 0
                })
        
        return detections

    def warmup(self):
        """
        더미 이미지로 워밍업
        """
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy)
        print("[MobileNet-SSD] Warmup complete")
