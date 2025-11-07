"""
Tracker Factory
설정에 따라 추적 모델 생성
"""

from .base import BaseTracker
from .botsort import BoTSORTTracker
from .bytetrack import ByteTrackTracker
from .deepsort import DeepSORTTracker


def get_tracker(name: str, **kwargs) -> BaseTracker:
    """
    추적 모델 인스턴스 반환
    
    Args:
        name: 모델 이름 ("botsort", "bytetrack", "deepsort")
        **kwargs: 모델별 추가 파라미터
    
    Returns:
        BaseTracker 인스턴스
    """
    name = name.lower()
    
    if name == "botsort":
        return BoTSORTTracker(**kwargs)
    elif name == "bytetrack":
        return ByteTrackTracker(**kwargs)
    elif name == "deepsort":
        return DeepSORTTracker(**kwargs)
    else:
        raise ValueError(f"Unknown tracker: {name}")


__all__ = [
    "BaseTracker",
    "BoTSORTTracker",
    "ByteTrackTracker",
    "DeepSORTTracker",
    "get_tracker",
]
