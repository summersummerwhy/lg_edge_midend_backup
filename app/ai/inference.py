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

    def update_face_pending(
        self,
        image: np.ndarray,
        tracks: List[Dict],
        entered_ids: List[int],
        n_sec: float = 0.5,
        m_try: int = 6,
        min_face_size: int = 80,
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
            log.info(
                "[FACE][TRY] track_id=%s attempt=%d/%d",
                tid, st["attempts"], m_try
            )
            st["next_time"] = now + n_sec

            box = t.get("box")
            if not box:
                continue

            x1, y1, x2, y2 = map(int, box)

            # 사람 박스 클램프
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            pw, ph = x2 - x1, y2 - y1
            if pw <= 0 or ph <= 0:
                continue
            if pw < min_face_size or ph < min_face_size:
                continue

            person_crop = image[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            faces = self.face_detector.detect_faces(person_crop)
            if not faces:
                log.info("[FACE][NOFACE] track_id=%s attempt=%d/%d", tid, st["attempts"], m_try)
                # 얼굴 못 찾으면 이번 시도 실패
                pass
            else:
                f = self.face_detector.select_main_face(faces)
                fx1, fy1, fx2, fy2 = map(int, f["box"])

                # 얼굴 박스도 crop 내부로 클램프 (중요)
                ch, cw = person_crop.shape[:2]
                fx1 = max(0, min(fx1, cw - 1))
                fx2 = max(0, min(fx2, cw))
                fy1 = max(0, min(fy1, ch - 1))
                fy2 = max(0, min(fy2, ch))

                fw, fh = fx2 - fx1, fy2 - fy1
                if fw >= min_face_size and fh >= min_face_size:
                    face_chip = person_crop[fy1:fy2, fx1:fx2]
                    if face_chip.size == 0:
                        log.info("[FACE][EMPTY_CROP] track_id=%s", tid)
                        continue
                    elif face_chip.size != 0:
                        try:
                            emb = self.face_embedder.embed(face_chip)
                            match = self.face_matcher.match(emb)  # matcher가 unknown 판정까지 하면 더 깔끔
                            face_id = match.get("id", "unknown")
                            score = float(match.get("score", 0.0))

                            log.info(
                                "[FACE][MATCH] track_id=%s id=%s score=%.3f (best=%s %.3f)",
                                tid,
                                face_id,
                                score,
                                st["best_id"],
                                st["best_score"],
                            )

                            # best 기록
                            if score > st["best_score"]:
                                st["best_score"] = score
                                st["best_id"] = face_id

                            # 성공 조건: unknown이 아니면 즉시 완료
                            if face_id != "unknown":
                                results.append({
                                    "type": "enter",
                                    "track_id": tid,
                                    "box": t.get("box"),
                                    "confidence": t.get("confidence", 0.0),
                                    "face_id": face_id,
                                    "face_score": score,
                                    "face_box": [x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2],
                                    "meta": {"attempts": st["attempts"], "mode": "face_retry"},
                                })
                                done_ids.append(tid)
                                log.info(
                                    "[FACE][SUCCESS] track_id=%s id=%s score=%.3f attempts=%d",
                                    tid, face_id, score, st["attempts"]
                                )
                                continue

                        except Exception as e:
                            log.exception("[FACE] pending embed/match error track_id=%s: %s", tid, e)

            # ---- 이번 시도에서 성공 못 했으면: m회 초과 시 unknown 완료 ----
            if st["attempts"] >= m_try:
                results.append({
                    "type": "enter",
                    "track_id": tid,
                    "box": t.get("box"),
                    "confidence": t.get("confidence", 0.0),
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


    
    # def run_face_recognition(
    #     self,
    #     image: np.ndarray,
    #     tracks: List[Dict],
    #     min_face_size: int = 80,
    #     match_threshold: float = 0.6,
    # ) -> List[Dict]:
    #     """
    #     person tracks + 전체 이미지 → 각 track별 얼굴 인식 결과 리스트.

    #     Args:
    #         image: 전체 프레임 (BGR, HWC)
    #         tracks: process_image() 결과 리스트
    #         min_face_size: 이 크기보다 작은 얼굴은 무시 (px)
    #         match_threshold: 이 값보다 similarity가 낮으면 "unknown" 처리

    #     Returns:
    #         [
    #             {
    #                 "track_id": 3,
    #                 "person_box": [x1,y1,x2,y2],
    #                 "face_box": [fx1,fy1,fx2,fy2],   # 전체 이미지 기준 좌표
    #                 "face_id": "Seohyun" or "unknown",
    #                 "face_score": 0.83,
    #             },
    #             ...
    #         ]
    #     """
    #     results: List[Dict] = []

    #     # 얼굴 파이프라인이 준비 안 된 경우
    #     if (
    #         getattr(self, "face_detector", None) is None
    #         or getattr(self, "face_embedder", None) is None
    #         or getattr(self, "face_matcher", None) is None
    #     ):
    #         return results

    #     h, w, _ = image.shape

    #     for t in tracks:
    #         box = t.get("box")
    #         track_id = t.get("track_id")
    #         if box is None or track_id is None:
    #             continue

    #         x1, y1, x2, y2 = map(int, box)
    #         # 이미지 바운더리 안으로 클램핑
    #         x1 = max(0, min(x1, w - 1))
    #         x2 = max(0, min(x2, w))
    #         y1 = max(0, min(y1, h - 1))
    #         y2 = max(0, min(y2, h))

    #         pw, ph = x2 - x1, y2 - y1
    #         if pw <= 0 or ph <= 0:
    #             continue

    #         # 사람 박스 자체가 너무 작으면 스킵
    #         if pw < min_face_size or ph < min_face_size:
    #             continue

    #         # 1) 사람 영역 crop
    #         person_crop = image[y1:y2, x1:x2]
    #         if person_crop.size == 0:
    #             continue

    #         # 2) 사람 영역 내부에서 얼굴 탐지
    #         faces = self.face_detector.detect_faces(person_crop)
    #         if not faces:
    #             continue

    #         # 가장 큰 얼굴 선택
    #         f = self.face_detector.select_main_face(faces)
    #         fx1, fy1, fx2, fy2 = map(int, f["box"])
    #         fw, fh = fx2 - fx1, fy2 - fy1

    #         if fw < min_face_size or fh < min_face_size:
    #             continue

    #         # person_crop 기준 얼굴 crop
    #         face_chip = person_crop[fy1:fy2, fx1:fx2]
    #         if face_chip.size == 0:
    #             continue

    #         # 3) embedding 추출
    #         try:
    #             emb = self.face_embedder.embed(face_chip)
    #         except Exception as e:
    #             log.exception("[FACE] embed error for track_id=%s: %s", track_id, e)
    #             continue

    #         # 4) DB 매칭
    #         match = self.face_matcher.match(emb)
    #         face_id = match.get("id", "unknown")
    #         score = float(match.get("score", 0.0))

    #         # 얼굴 박스를 전체 이미지 기준으로 변환
    #         abs_fx1 = x1 + fx1
    #         abs_fy1 = y1 + fy1
    #         abs_fx2 = x1 + fx2
    #         abs_fy2 = y1 + fy2

    #         results.append(
    #             {
    #                 "track_id": track_id,
    #                 "person_box": [x1, y1, x2, y2],
    #                 "face_box": [abs_fx1, abs_fy1, abs_fx2, abs_fy2],
    #                 "face_id": face_id,
    #                 "face_score": score,
    #             }
    #         )

    #     return results

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


