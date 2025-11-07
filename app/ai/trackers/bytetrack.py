"""
ByteTrack Tracker (BoT-SORT보다 2배 빠름)
간단한 IoU 기반 추적
"""

import numpy as np
from typing import List, Dict
from .base import BaseTracker


def iou(box1, box2):
    """
    IoU (Intersection over Union) 계산
    box: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


class ByteTrackTracker(BaseTracker):
    """
    ByteTrack 추적 모델
    - BoT-SORT보다 2배 빠름
    - IoU 기반 간단한 매칭
    - 정확도는 BoT-SORT와 비슷
    """

    def __init__(self, track_buffer: int = 30, match_thresh: float = 0.8):
        super().__init__()
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.tracks = {}  # {track_id: {"box": [...], "confidence": ..., "age": ...}}
        self.next_id = 1

    def update(self, detections: List[Dict], image: np.ndarray = None) -> List[Dict]:
        """
        감지 결과를 받아 추적 수행
        """
        # 기존 track age 증가
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1
            # 오래된 track 제거
            if self.tracks[tid]["age"] > self.track_buffer:
                del self.tracks[tid]
                del self.track_history[tid]
        
        # Detection과 기존 track 매칭
        matched_tracks = []
        unmatched_dets = []
        
        if not detections:
            self.frame_count += 1
            return []
        
        if not self.tracks:
            # 첫 프레임: 모든 detection을 새 track으로
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
            # IoU 기반 매칭
            det_matched = [False] * len(detections)
            track_matched = {tid: False for tid in self.tracks.keys()}
            
            for i, det in enumerate(detections):
                best_iou = 0
                best_tid = None
                
                for tid, track in self.tracks.items():
                    if track_matched[tid]:
                        continue
                    
                    iou_score = iou(det["box"], track["box"])
                    if iou_score > best_iou and iou_score > self.match_thresh:
                        best_iou = iou_score
                        best_tid = tid
                
                if best_tid is not None:
                    # 매칭 성공
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
            
            # 매칭 안 된 detection → 새 track
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
        print("[ByteTrack] Reset complete")
