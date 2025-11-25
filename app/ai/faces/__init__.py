"""
Face 관련 factory 함수들

- get_face_matcher(name)
"""

from typing import Optional
from pathlib import Path

from .base import BaseFaceMatcher




def get_face_matcher(name: Optional[str]) -> Optional[BaseFaceMatcher]:
    """
    embedding 기반 매칭기 생성.
    """
    if name is None:
        return None

    lname = name.lower()

    # TODO
    if lname == "simple":
        from .simple_matcher import SimpleFaceMatcher

        return SimpleFaceMatcher()
    else:
        raise ValueError(f"Unknown face matcher: {name}")
