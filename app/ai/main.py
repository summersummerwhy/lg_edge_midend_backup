"""
기존 코드와 호환되는 main.py
track_image_by_path, track_image 함수 제공
"""

import logging
import cv2
import numpy as np
import base64
from pathlib import Path
from typing import List, Dict

from .inference import get_ai_instance

log = logging.getLogger(__name__)

# 전역 track_list (기존 코드 호환용)
track_list = {}


def _draw_boxes_on_image(image: np.ndarray, tracks: List[Dict], event_type: str) -> np.ndarray:
    """
    이미지에 박스 그리기
    
    Args:
        image: 원본 이미지
        tracks: 추적 결과
        event_type: "enter" 또는 "exit"
    
    Returns:
        박스가 그려진 이미지
    """
    img_copy = image.copy()
    
    # 색상 선택
    color = (0, 255, 0) if event_type == "enter" else (0, 0, 255)  # Green / Red
    
    for track in tracks:
        box = track["box"]
        track_id = track["track_id"]
        
        # 박스 그리기
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # track_id 표시
        label = f"ID:{track_id}"
        cv2.putText(img_copy, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img_copy


def track_image_by_path(image_path: Path) -> List[Dict]:
    """
    이미지 파일 경로로 추적 수행 (기존 코드 호환)
    
    Args:
        image_path: 이미지 파일 경로
    
    Returns:
        MQTT payload 리스트
        [
            {
                "type": "enter" or "exit",
                "track_id": 1,
                "timestamp": "...",
                "image": {
                    "format": "jpg",
                    "width": 640,
                    "height": 480,
                    "data_b64": "..."
                }
            },
            ...
        ]
    """
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        log.error(f"[AI] Failed to load image: {image_path}")
        return []
    
    return track_image(image, "jpg")


def track_image(image: np.ndarray, format: str = "jpg") -> List[Dict]:
    """
    이미지 배열로 추적 수행 (기존 코드 호환)
    
    Args:
        image: OpenCV 이미지 (HWC, BGR)
        format: 이미지 포맷 ("jpg", "png" 등)
    
    Returns:
        MQTT payload 리스트
    """
    global track_list
    
    ai = get_ai_instance()
    
    # 1. Detection + Tracking
    tracks = ai.process_image(image)
    
    # 2. 입장/퇴장 감지
    events = ai.detect_enter_exit(tracks)
    
    entered_ids = events["entered"]
    exited_ids = events["exited"]
    track_list = events["current"]
    
    # 3. Payload 생성
    payloads = []
    h, w = image.shape[:2]
    
    # 입장 이벤트
    for track_id in entered_ids:
        # 해당 track 찾기
        track = next((t for t in tracks if t["track_id"] == track_id), None)
        if track is None:
            continue
        
        # 박스 그리기
        img_with_box = _draw_boxes_on_image(image, [track], "enter")
        
        # Base64 인코딩
        _, buffer = cv2.imencode(f'.{format}', img_with_box)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        
        payload = {
            "type": "enter",
            "track_id": track_id,
            "box": track["box"],
            "confidence": track["confidence"],
            "image": {
                "format": format,
                "width": w,
                "height": h,
                "data_b64": img_b64
            }
        }
        payloads.append(payload)
        log.info(f"[AI] ENTER: track_id={track_id}")
    
    # 퇴장 이벤트
    for track_id in exited_ids:
        # 퇴장은 이미지 없음 (이미 사라짐)
        payload = {
            "type": "exit",
            "track_id": track_id,
        }
        payloads.append(payload)
        log.info(f"[AI] EXIT: track_id={track_id}")
    
    return payloads