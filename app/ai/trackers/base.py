"""
Base Tracker 클래스
모든 추적 모델이 상속받는 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class BaseTracker(ABC):
    """
    모든 추적 모델의 기본 인터페이스
    """

    def __init__(self):
        self.track_history = {}  # {track_id: last_seen_frame}
        self.frame_count = 0

    @abstractmethod
    def update(self, detections: List[Dict], image: np.ndarray = None) -> List[Dict]:
        """
        감지 결과를 받아 추적 수행
        
        Args:
            detections: Detector가 반환한 감지 결과
                [
                    {
                        "box": [x1, y1, x2, y2],
                        "confidence": 0.95,
                        "class_id": 0
                    },
                    ...
                ]
            image: 원본 이미지 (일부 tracker에서 필요)
        
        Returns:
            추적 결과:
            [
                {
                    "box": [x1, y1, x2, y2],
                    "track_id": 1,
                    "confidence": 0.95
                },
                ...
            ]
        """
        pass

    @abstractmethod
    def reset(self):
        """
        추적 상태 초기화
        """
        pass

    def get_active_tracks(self) -> List[int]:
        """
        현재 활성화된 track_id 목록 반환
        """
        return list(self.track_history.keys())
