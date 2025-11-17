"""
Detector Factory
설정에 따라 감지 모델 생성
"""

from .base import BaseDetector
from .yolov11n import YOLO11nDetector
from .yolov8n import YOLOv8nDetector
from .yolov5n import YOLOv5nDetector



def get_detector(name: str, confidence: float = 0.5) -> BaseDetector:
    """
    감지 모델 인스턴스 반환
    
    Args:
        name: 모델 이름 ("yolov11n", "yolov8n", "yolov5n")
        confidence: 감지 임계값
    
    Returns:
        BaseDetector 인스턴스
    """
    name = name.lower()
    
    if name == "yolov11n":
        return YOLO11nDetector(confidence=confidence)
    elif name == "yolov8n":
        return YOLOv8nDetector(confidence=confidence)
    elif name == "yolov5n":  
        return YOLOv5nDetector(confidence=confidence)
    else:
        raise ValueError(f"Unknown detector: {name}")


__all__ = [
    "BaseDetector",
    "YOLO11nDetector",
    "YOLOv8nDetector",
    "MediaPipeDetector",
    "get_detector",
]
