"""
통합 Inference 모듈
Detector + Tracker 조합
"""

import time
import logging
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import cv2

from .detectors import get_detector, BaseDetector
from .trackers import get_tracker, BaseTracker
from .ai_config import DETECTOR, TRACKER, CONFIDENCE_THRESHOLD, BENCHMARK_MODE, ARCFACE_ONNX_PATH
from .faces import get_face_matcher
from .faces.detector import MediaPipeFaceDetector
from .faces.embedder import ArcFaceONNXEmbedder
from .faces.simple_matcher import SimpleFaceMatcher

log = logging.getLogger(__name__)


class AIInference:
    """
    AI 추론 파이프라인
    Person(Detector + Tracker) + Face(Detector + Embedder + Matcher)  조합
    """

    def __init__(
        self, 
        detector_name: str = None, 
        tracker_name: str = None,
    ):
        """
        Args:
            detector_name: 감지 모델 이름 (None이면 config에서 로드)
            tracker_name: 추적 모델 이름 (None이면 config에서 로드)
        """
        self.detector_name = detector_name or DETECTOR
        self.tracker_name = tracker_name or TRACKER

        
        log.info(f"[AI] Initializing {self.detector_name} + {self.tracker_name}")
        
        # 모델 로드
        self.detector: BaseDetector = get_detector(
            self.detector_name, 
            confidence=CONFIDENCE_THRESHOLD
        )
        self.tracker: BaseTracker = get_tracker(self.tracker_name)

        # face pipleline load
        self.face_detector = MediaPipeFaceDetector()
        self.face_embedder = ArcFaceONNXEmbedder(
            model_path=ARCFACE_ONNX_PATH,
            input_size=(112, 112),
            normalize=True,
        )
        self.face_matcher = SimpleFaceMatcher()
        
        # 워밍업
        self.detector.warmup()
        
        # 이전 track 상태 (입장/퇴장 감지용)
        self.previous_tracks = {}  # {track_id: box}
        
        # FPS 측정
        self.fps_history = []
        
        log.info(f"[AI] Ready: {self.detector_name} + {self.tracker_name}")

    def process_image(self, image: np.ndarray) -> List[Dict]:
        """
        이미지 처리 (Detection + Tracking)
        
        Args:
            image: OpenCV 이미지 (HWC, BGR)
        
        Returns:
            추적 결과 리스트
            [
                {
                    "box": [x1, y1, x2, y2],
                    "track_id": 1,
                    "confidence": 0.95
                },
                ...
            ]
        """
        start_time = time.time()
        
        # 1. Detection
        detections = self.detector.detect(image)
        
        # 2. Tracking
        # BoT-SORT는 이미지 필요
        if self.tracker_name == "botsort":
            tracks = self.tracker.update(detections, image=image)
        else:
            tracks = self.tracker.update(detections)
        
        # FPS 측정
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        
        if BENCHMARK_MODE:
            self.fps_history.append(fps)
            if len(self.fps_history) % 10 == 0:
                avg_fps = sum(self.fps_history[-10:]) / 10
                log.info(f"[BENCHMARK] FPS: {fps:.2f} (avg: {avg_fps:.2f})")
        
        return tracks

    def detect_enter_exit(self, tracks: List[Dict]) -> Dict:
        """
        입장/퇴장 감지
        
        Args:
            tracks: process_image() 결과
        
        Returns:
            {
                "entered": [track_id, ...],  # 새로 나타난 ID
                "exited": [track_id, ...],   # 사라진 ID
                "current": {track_id: box}   # 현재 추적 중
            }
        """
        current_ids = {t["track_id"] for t in tracks}
        previous_ids = set(self.previous_tracks.keys())
        
        entered = list(current_ids - previous_ids)
        exited = list(previous_ids - current_ids)
        
        # 현재 상태 저장
        current_tracks = {t["track_id"]: t["box"] for t in tracks}
        self.previous_tracks = current_tracks
        
        return {
            "entered": entered,
            "exited": exited,
            "current": current_tracks
        }

    def get_fps_stats(self) -> Dict:
        """
        FPS 통계 반환
        """
        if not self.fps_history:
            return {"avg": 0, "min": 0, "max": 0}
        
        return {
            "avg": sum(self.fps_history) / len(self.fps_history),
            "min": min(self.fps_history),
            "max": max(self.fps_history),
            "count": len(self.fps_history)
        }

    def reset(self):
        """
        추적 상태 초기화
        """
        self.tracker.reset()
        self.previous_tracks.clear()
        self.fps_history.clear()
        log.info("[AI] Reset complete")


# 전역 인스턴스 (싱글톤)
_ai_instance: Optional[AIInference] = None


def get_ai_instance(detector: str = None, tracker: str = None) -> AIInference:
    """
    AI 인스턴스 반환 (싱글톤)
    """
    global _ai_instance
    
    if _ai_instance is None:
        _ai_instance = AIInference(detector, tracker)
    
    return _ai_instance


def process_image_file(image_path: Path) -> List[Dict]:
    """
    이미지 파일 처리
    
    Args:
        image_path: 이미지 파일 경로
    
    Returns:
        추적 결과
    """
    ai = get_ai_instance()
    image = cv2.imread(str(image_path))
    
    if image is None:
        log.error(f"[AI] Failed to load image: {image_path}")
        return []
    
    return ai.process_image(image)


def process_image_array(image: np.ndarray) -> List[Dict]:
    """
    이미지 배열 처리
    
    Args:
        image: OpenCV 이미지 (HWC, BGR)
    
    Returns:
        추적 결과
    """
    ai = get_ai_instance()
    return ai.process_image(image)
