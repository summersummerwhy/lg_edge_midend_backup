"""
Detector Factory
설정에 따라 감지 모델 생성
"""

from .base import BaseDetector
from .yolo11n import YOLO11nDetector
from .yolov8n import YOLOv8nDetector
from .mobilenet_ssd import MobileNetSSDDetector


def get_detector(name: str, confidence: float = 0.5) -> BaseDetector:
    """
    감지 모델 인스턴스 반환
    
    Args:
        name: 모델 이름 ("yolo11n", "yolov8n", "mobilenet_ssd")
        confidence: 감지 임계값
    
    Returns:
        BaseDetector 인스턴스
    """
    name = name.lower()
    
    if name == "yolo11n":
        return YOLO11nDetector(confidence=confidence)
    elif name == "yolov8n":
        return YOLOv8nDetector(confidence=confidence)
    elif name == "mobilenet_ssd":
        return MobileNetSSDDetector(confidence=confidence)
    else:
        raise ValueError(f"Unknown detector: {name}")


__all__ = [
    "BaseDetector",
    "YOLO11nDetector",
    "YOLOv8nDetector",
    "MobileNetSSDDetector",
    "get_detector",
]
