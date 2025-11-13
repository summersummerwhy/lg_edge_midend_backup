"""
DeepSORT Tracker (중간 성능)
ByteTrack보다 정확하지만 느림
"""

import numpy as np
from typing import List, Dict
from .base import BaseTracker


class DeepSORTTracker(BaseTracker):
    """
    DeepSORT 추적 모델
    - ByteTrack와 BoT-SORT 중간 성능
    - 외관 특징(appearance feature) 사용
    - 현재는 간단한 IoU 버전 (실제 DeepSORT는 feature extractor 필요)
    """

    def __init__(self, track_buffer: int = 30, match_thresh: float = 0.7):
        super().__init__()
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.tracks = {}
        self.next_id = 1

    def _iou(self, box1, box2):
        """IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0

    def update(self, detections: List[Dict], image: np.ndarray = None) -> List[Dict]:
        """
        감지 결과를 받아 추적 수행
        
        참고: 실제 DeepSORT는 appearance feature를 사용하지만
              여기서는 간단하게 IoU만 사용 (ByteTrack과 유사)
        """
        # 기존 track age 증가
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.track_buffer:
                del self.tracks[tid]
                del self.track_history[tid]
        
        matched_tracks = []
        
        if not detections:
            self.frame_count += 1
            return []
        
        if not self.tracks:
            # 첫 프레임
            for det in detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "box": det["box"],
                    "confidence": det["confidence"],
                    "age": 0
                }
                self.track_history[tid] = self.frame_count
                matched_tracks.append({
                    "box": det["box"],
                    "track_id": tid,
                    "confidence": det["confidence"]
                })
        else:
            # 매칭
            det_matched = [False] * len(detections)
            track_matched = {tid: False for tid in self.tracks.keys()}
            
            # 1차: 높은 confidence detection부터 매칭
            sorted_dets = sorted(enumerate(detections), 
                               key=lambda x: x[1]["confidence"], 
                               reverse=True)
            
            for i, det in sorted_dets:
                best_iou = 0
                best_tid = None
                
                for tid, track in self.tracks.items():
                    if track_matched[tid]:
                        continue
                    
                    iou_score = self._iou(det["box"], track["box"])
                    if iou_score > best_iou and iou_score > self.match_thresh:
                        best_iou = iou_score
                        best_tid = tid
                
                if best_tid is not None:
                    det_matched[i] = True
                    track_matched[best_tid] = True
                    self.tracks[best_tid]["box"] = det["box"]
                    self.tracks[best_tid]["confidence"] = det["confidence"]
                    self.tracks[best_tid]["age"] = 0
                    self.track_history[best_tid] = self.frame_count
                    matched_tracks.append({
                        "box": det["box"],
                        "track_id": best_tid,
                        "confidence": det["confidence"]
                    })
            
            # 새 track 생성
            for i, det in enumerate(detections):
                if not det_matched[i]:
                    tid = self.next_id
                    self.next_id += 1
                    self.tracks[tid] = {
                        "box": det["box"],
                        "confidence": det["confidence"],
                        "age": 0
                    }
                    self.track_history[tid] = self.frame_count
                    matched_tracks.append({
                        "box": det["box"],
                        "track_id": tid,
                        "confidence": det["confidence"]
                    })
        
        self.frame_count += 1
        return matched_tracks

    def reset(self):
        """
        추적 상태 초기화
        """
        self.tracks.clear()
        self.track_history.clear()
        self.next_id = 1
        self.frame_count = 0
        print("[DeepSORT] Reset complete")
