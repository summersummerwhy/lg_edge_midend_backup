"""
Base Detector 클래스
모든 감지 모델이 상속받는 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class BaseDetector(ABC):
    """
    모든 감지 모델의 기본 인터페이스
    """

    def __init__(self, confidence: float = 0.5):
        self.confidence = confidence

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        이미지에서 사람 감지
        
        Args:
            image: OpenCV 이미지 (HWC, BGR)
        
        Returns:
            List of detections:
            [
                {
                    "box": [x1, y1, x2, y2],  # 좌상단, 우하단
                    "confidence": 0.95,
                    "class_id": 0  # person
                },
                ...
            ]
        """
        pass

    @abstractmethod
    def warmup(self):
        """
        모델 워밍업 (첫 추론 속도 개선)
        """
        pass

    def filter_persons(self, detections: List[Dict]) -> List[Dict]:
        """
        사람(class_id=0)만 필터링
        """
        return [det for det in detections if det.get("class_id") == 0]
