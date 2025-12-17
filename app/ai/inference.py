"""
통합 Inference 모듈
Detector + Tracker + Face 작업들 조합
"""

import time
import logging
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import cv2

from .detectors import get_detector, BaseDetector
from .ai_config import (DETECTOR, 
    TRACKER, 
    CONFIDENCE_THRESHOLD, 
    BENCHMARK_MODE, 
    ARCFACE_ONNX_PATH,     
    FACE_RETRY_INTERVAL_SEC,
    FACE_MAX_TRY,
    FACE_MIN_SIZE
)
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

        # face pipleline load
        self.face_detector = MediaPipeFaceDetector()
        self.face_embedder = ArcFaceONNXEmbedder(
            model_path=ARCFACE_ONNX_PATH,
            input_size=(112, 112),
            normalize=True,
        )
        self.face_matcher = SimpleFaceMatcher(threshold=0.54)
        
        # 워밍업
        self.detector.warmup()
        
        # 이전 track 상태 (입장/퇴장 감지용)
        self.previous_tracks = {}  # {track_id: box}
        self.face_pending: Dict[int, Dict] = {}
        self.entered_tracks = set()
        self.track_face_map = {}
        
        # FPS 측정
        self.fps_history = []
        
        log.info(f"[AI] Ready: {self.detector_name} + {self.tracker_name}")

    
    def process_image(self, image: np.ndarray) -> List[Dict]:
        """
        이미지 처리 (YOLO internal Tracking: BoT-SORT)
        Returns:
            [
              {"box":[x1,y1,x2,y2], "track_id": 1, "confidence": 0.95},
              ...
            ]
        """
        start_time = time.time()

        # detector 내부 YOLO 모델 꺼내기
        yolo = getattr(self.detector, "model", None)
        if yolo is None or not hasattr(yolo, "track"):
            raise RuntimeError(
                "[AI] detector.model.track()를 찾을 수 없음. "
                "YOLOv8nDetector가 self.model = YOLO(...) 형태인지 확인해줘."
            )

        # tracker yaml 선택
        tracker_yaml = "botsort.yaml" if self.tracker_name == "botsort" else "bytetrack.yaml"

        # YOLO 내장 tracker 실행 (persist=True 중요!)
        results = yolo.track(
            source=image,
            persist=True,
            verbose=False,
            conf=CONFIDENCE_THRESHOLD,
            tracker=tracker_yaml,
            classes=[0],  # person만 (COCO)
        )

        tracks: List[Dict] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i].tolist()
                # ids가 없으면 트래킹 ID가 없다는 뜻 -> 이 박스는 스킵
                if ids is None:
                    continue

                tid = int(ids[i])
                # 혹시 음수면(거의 없음) 스킵
                if tid < 0:
                    continue

                tracks.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "track_id": tid,
                    "confidence": float(confs[i]) if confs is not None else 0.0,
                })

        # FPS
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        if BENCHMARK_MODE:
            self.fps_history.append(fps)
            if len(self.fps_history) % 10 == 0:
                avg_fps = sum(self.fps_history[-10:]) / 10
                log.info(f"[BENCHMARK] FPS: {fps:.2f} (avg: {avg_fps:.2f})")

        return tracks

    @staticmethod
    def _clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        return x1, y1, x2, y2


    def update_face_pending(
        self,
        image: np.ndarray,
        tracks: List[Dict],
        entered_ids: List[int],
    ) -> List[Dict]:
        """
        입장한 track_id들에 대해 n초 간격으로 최대 m번 얼굴 인식 시도.
        성공하면 바로 'enter+face' payload용 dict 반환.
        m번까지 실패하면 unknown으로 enter 반환.
        """

        now = time.time()
        results: List[Dict] = []

        # 현재 프레임의 track_id -> track dict 맵
        track_map = {t.get("track_id"): t for t in tracks if t.get("track_id") is not None}

        # 1) 새로 들어온 id를 pending에 등록 (아직 enter는 안 보냄)
        for tid in entered_ids:
            log.info("[FACE][PENDING] start track_id=%s", tid)
            if tid not in self.face_pending:
                self.face_pending[tid] = {
                    "attempts": 0,
                    "next_time": now,      # 바로 1회 시도 가능
                    "first_time": now,
                    "best_id": "unknown",
                    "best_score": -1.0,
                }

        # 2) pending 중에서, 현재 프레임에 존재하고 시도 시간이 된 것만 처리
        done_ids = []

        h, w = image.shape[:2]

        for tid, st in list(self.face_pending.items()):
            # 아직 화면에 없으면(트래킹 끊김/아직 미검출) 다음 프레임로
            t = track_map.get(tid)
            if t is None:
                continue

            if now < st["next_time"]:
                continue

            # ---- 얼굴 인식 1회 시도 ----
            st["attempts"] += 1
            st["next_time"] = now + FACE_RETRY_INTERVAL_SEC
            log.info(
                "[FACE][TRY] track_id=%s attempt=%d/%d",
                tid, st["attempts"], FACE_MAX_TRY
            )

            try:
                r = self.try_recognize_face_on_track(
                    image=image,
                    track=t,
                )
            except Exception as e:
                log.exception("[FACE] try_recognize_face_on_track error track_id=%s: %s", tid, e)
                r = None

            if r is None:
                log.info("[FACE][NOFACE] track_id=%s attempt=%d/%d", tid, st["attempts"], FACE_MAX_TRY)
            else:
                face_id = r["face_id"]
                score = float(r["face_score"])

                # best 기록
                if score > st["best_score"]:
                    st["best_score"] = score
                    st["best_id"] = face_id

                log.info(
                    "[FACE][MATCH] track_id=%s id=%s score=%.3f (best=%s %.3f)",
                    tid, face_id, score, st["best_id"], st["best_score"]
                )

                # 성공 조건: unknown이 아니면 즉시 완료
                if str(face_id).lower() != "unknown":
                    self.track_face_map[tid] = face_id
                    results.append({
                        "type": "enter",
                        "track_id": tid,
                        "face_id": face_id,
                        "face_score": score,
                        "face_box": r.get("face_box"),
                        "meta": {"attempts": st["attempts"], "mode": "face_retry"},
                    })
                    done_ids.append(tid)
                    log.info("[FACE][SUCCESS] track_id=%s id=%s score=%.3f", tid, face_id, score)
                    continue

            # ---- m회 초과 시 unknown 완료 ----
            if st["attempts"] >= FACE_MAX_TRY:
                self.track_face_map[tid] = "unknown"
                results.append({
                    "type": "enter",
                    "track_id": tid,
                    "face_id": "unknown",
                    "face_score": float(st["best_score"]) if st["best_score"] >= 0 else 0.0,
                    "face_box": None,
                    "meta": {"attempts": st["attempts"], "mode": "face_retry", "reason": "max_tries"},
                })
                done_ids.append(tid)
                log.info(
                    "[FACE][FAIL] track_id=%s -> unknown after %d attempts (best=%s %.3f)",
                    tid, st["attempts"], st["best_id"], st["best_score"]
                )

        # 3) 완료된 pending 제거
        for tid in set(done_ids):
            self.face_pending.pop(tid, None)

        return results

    
    
    def try_recognize_face_on_track(
        self,
        image: np.ndarray,
        track: Dict,
    ) -> Optional[Dict]:
        """
        단일 track에 대해 얼굴 인식 1회 시도.
        성공/실패 여부와 관계없이 "이번 시도 결과"를 반환하거나,
        인식할 조건이 안 되면 None 반환.

        Returns (성공 시):
            {
              "face_id": str,
              "face_score": float,
              "face_box": [abs_fx1, abs_fy1, abs_fx2, abs_fy2],
            }

        Returns (실패/미검출 시):
            None
        """
        # 얼굴 파이프라인 준비 안 된 경우
        if (
            getattr(self, "face_detector", None) is None
            or getattr(self, "face_embedder", None) is None
            or getattr(self, "face_matcher", None) is None
        ):
            return None

        box = track.get("box")
        if not box:
            return None

        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1, y1, x2, y2 = self._clamp_box(x1, y1, x2, y2, w, h)

        pw, ph = x2 - x1, y2 - y1
        if pw <= 0 or ph <= 0:
            return None

        person_crop = image[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None

        # 사람 영역 내부에서 얼굴 탐지
        faces = self.face_detector.detect_faces(person_crop)
        if not faces:
            return None

        f = self.face_detector.select_main_face(faces)
        fx1, fy1, fx2, fy2 = map(int, f["box"])

        ch, cw = person_crop.shape[:2]
        fx1, fy1, fx2, fy2 = self._clamp_box(fx1, fy1, fx2, fy2, cw, ch)

        fw, fh = fx2 - fx1, fy2 - fy1
        if fw < FACE_MIN_SIZE or fh < FACE_MIN_SIZE:
            return None

        face_chip = person_crop[fy1:fy2, fx1:fx2]
        if face_chip.size == 0:
            return None

        # embedding + match
        emb = self.face_embedder.embed(face_chip)
        match = self.face_matcher.match(emb)

        face_id = match.get("id", "unknown")
        score = float(match.get("score", 0.0))

        abs_box = [x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2]
        return {"face_id": face_id, "face_score": score, "face_box": abs_box}


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
        exited_ids = list(previous_ids - current_ids)
        
        exited = []
        for tid in exited_ids:
            face_id = self.track_face_map.pop(tid, "unknown")
            box = self.previous_tracks.get(tid)
            exited.append({
                "track_id": tid,
                "box": box,
                "face_id": face_id
            })
        
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
        self.previous_tracks.clear()
        self.fps_history.clear()
    
        # best-effort: ultralytics predictor tracker reset 시도
        yolo = getattr(self.detector, "model", None)
        try:
            if hasattr(yolo, "predictor") and hasattr(yolo.predictor, "tracker") and yolo.predictor.tracker is not None:
                yolo.predictor.tracker.reset()
        except Exception:
            pass
    
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


