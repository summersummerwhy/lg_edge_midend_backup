"""
Face 관련 factory 함수들

- get_face_detector(name)
- get_face_embedder(name)
- get_face_matcher(name)
"""

from typing import Optional

from .base import BaseFaceDetector, BaseFaceEmbedder, BaseFaceMatcher


def get_face_detector(name: Optional[str]) -> Optional[BaseFaceDetector]:
    """
    얼굴 detector 인스턴스 생성.

    Args:
        name: "mediapipe", "yunet" 등. None이면 사용 안 함.

    Returns:
        BaseFaceDetector or None
    """
    if name is None:
        return None

    lname = name.lower()

    # TODO: 
    # if lname == "mediapipe":
    #     from .mediapipe_detector import MediaPipeFaceDetector

    #     return MediaPipeFaceDetector()
    # elif lname == "yunet":
    #     from .yunet_detector import YuNetFaceDetector

    #     return YuNetFaceDetector()
    # else:
    #     raise ValueError(f"Unknown face detector: {name}")


def get_face_embedder(name: Optional[str]) -> Optional[BaseFaceEmbedder]:
    """
    얼굴 embedding 추출기 생성.
    """
    if name is None:
        return None

    lname = name.lower()

    # TODO: 
    # if lname == "dummy":
    #     from .dummy_embedder import DummyFaceEmbedder

    #     return DummyFaceEmbedder()
    # elif lname == "arcface":
    #     from .arcface_embedder import ArcFaceEmbedder

    #     return ArcFaceEmbedder()
    # else:
    #     raise ValueError(f"Unknown face embedder: {name}")


def get_face_matcher(name: Optional[str]) -> Optional[BaseFaceMatcher]:
    """
    embedding 기반 매칭기 생성.
    """
    if name is None:
        return None

    lname = name.lower()

    # TODO
    # if lname == "simple":
    #     from .simple_matcher import SimpleFaceMatcher

    #     return SimpleFaceMatcher()
    # else:
    #     raise ValueError(f"Unknown face matcher: {name}")
