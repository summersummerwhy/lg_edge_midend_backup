"""
BoT-SORT Tracker (현재 사용 중)
YOLO 내장 BoT-SORT 사용
"""

import numpy as np
from typing import List, Dict
from .base import BaseTracker
from ultralytics import YOLO


class BoTSORTTracker(BaseTracker):
    """
    BoT-SORT 추적 모델
    - 높은 정확도
    - YOLO 내장 tracker 사용
    - 속도는 느린 편
    """

    def __init__(self, model_path: str = "yolo11n.pt"):
        super().__init__()
        self.model = YOLO(model_path)
        print("[BoT-SORT] Loaded with YOLO tracker")

    def update(self, detections: List[Dict], image: np.ndarray = None) -> List[Dict]:
        """
        감지 결과를 받아 추적 수행
        
        참고: BoT-SORT는 YOLO.track()과 통합되어 있어
              실제로는 이 메서드가 아닌 YOLO.track()을 직접 사용하는 게 더 효율적
        """
        if image is None:
            raise ValueError("BoT-SORT requires image input")
        
        # YOLO track 사용 (detection + tracking 통합)
        results = self.model.track(image, persist=True, verbose=False)
        
        tracks = []
        if results and results[0].boxes is not None and results[0].boxes.is_track:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, tid, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
                if cls_id == 0:  # person만
                    tracks.append({
                        "box": box.tolist(),
                        "track_id": int(tid),
                        "confidence": float(conf)
                    })
                    self.track_history[int(tid)] = self.frame_count
        
        self.frame_count += 1
        return tracks

    def reset(self):
        """
        추적 상태 초기화
        """
        self.track_history.clear()
        self.frame_count = 0
        # YOLO tracker 리셋
        self.model.predictor = None
        print("[BoT-SORT] Reset complete")
