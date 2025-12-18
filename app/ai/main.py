
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
    global track_list

    ai = get_ai_instance()

    # 1) Detection + Tracking
    tracks = ai.process_image(image)

    # 2) 입장/퇴장 감지
    events = ai.detect_enter_exit(tracks)

    entered_ids = events["entered"]
    exited_ids = events["exited"]
    track_list = events["current"]

    payloads: List[Dict] = []
    h, w = image.shape[:2]

    # 3) 퇴장 이벤트는 즉시 전송 + pending 정리
    for exit_item in exited_ids:
        track_id = exit_item["track_id"]
        face_id = exit_item.get("face_id", "unknown")
        box = exit_item.get("box")

        # pending에 남아있던 입장 대기(track)도 정리
        if hasattr(ai, "face_pending"):
            ai.face_pending.pop(track_id, None)

        # enter 보낸 적 없는 track이면 exit 드랍
        if hasattr(ai, "entered_tracks") and track_id not in ai.entered_tracks:
            log.info(f"[AI] DROP EXIT(no enter): track_id={track_id}")
            continue

        payloads.append({
            "type": "exit",
            "track_id": track_id,
            "face_id": face_id,
            "box": box,
        })
        log.info(f"[AI] EXIT: track_id={track_id} face_id={face_id}")

        # exit 보냈으면 기록에서 제거(메모리/재사용 대비)
        if hasattr(ai, "entered_tracks"):
            ai.entered_tracks.discard(track_id)

    # 4) 입장 이벤트: 얼굴 재시도 로직이 "끝난 것만" enter로 전송
    # TODO: 키패드 누르는 시간 정도로 설정...
    #    n=0.5초 간격, m=6번 시도 (총 ~3초) 
    try:
        enter_results = ai.update_face_pending(
            image=image,
            tracks=tracks,
            entered_ids=entered_ids,
        )
    except AttributeError:
        # update_face_pending을 아직 안 붙였으면 아무것도 안 보냄
        enter_results = []
        log.debug("[AI] update_face_pending not implemented, skipping enter(face-retry)")
    except Exception as e:
        enter_results = []
        log.exception("[AI] Face pending error: %s", e)

    # enter_results는 "확정된 enter"만 들어있음
    # -> 성공이면 face_id != unknown, 실패면 face_id == unknown
    for er in enter_results:
        track_id = er["track_id"]

        # 박스 그린 이미지 포함시키고 싶으면 여기서 생성
        track = next((t for t in tracks if t.get("track_id") == track_id), None)
        if track is None:
            continue

        img_with_box = _draw_boxes_on_image(image, [track], "enter")

        _, buffer = cv2.imencode(f".{format}", img_with_box)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        payloads.append({
            "type": "enter",
            "track_id": track_id,
            "box": track.get("box"),
            "confidence": track.get("confidence", 0.0),

            # 얼굴 확정 결과
            "face_id": er.get("face_id", "unknown"),
            "face_score": er.get("face_score", 0.0),
            "face_box": er.get("face_box"),   # 없으면 None

            # 디버깅용 메타 (attempts 등)
            "meta": er.get("meta", {}),

            "image": {
                "format": format,
                "width": w,
                "height": h,
                "data_b64": img_b64,
            },
        })

        if hasattr(ai, "entered_tracks"):
            ai.entered_tracks.add(track_id)

        log.info(
            "[AI] ENTER(resolved): track_id=%s face_id=%s score=%.3f tries=%s",
            track_id,
            payloads[-1]["face_id"],
            float(payloads[-1]["face_score"]),
            payloads[-1]["meta"].get("attempts"),
        )

    # payloads = convert_enter_exit(payloads)

    return payloads

ENTER_THRESHOLD = 0
EXIT_THRESHOLD = 0

def convert_enter_exit(payloads: List[Dict]) -> List[Dict]:
    new_payloads = []

    # box = x1,y1,x2,y2

    for payload in payloads:
        if payload["type"] == "enter":
            # 예시
            if payload["box"][1] >= ENTER_THRESHOLD:
                payload["type"] = "exit"
                new_payloads.append(payload)
        elif payload["type"] == "exit":
            # 예시
            if payload["box"][1] <= EXIT_THRESHOLD:
                payload["type"] = "enter"
                new_payloads.append(payload)

    return new_payloads