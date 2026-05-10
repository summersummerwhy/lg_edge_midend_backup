"""
웹캠으로 얼굴 여러 장 캡처해서 embeddings.json에 등록하는 임시 CLI

사용 예시:
    python -m app.ai.faces.register_face --id Alice
"""

import argparse
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .detector import MediaPipeFaceDetector
from .embedder import ArcFaceONNXEmbedder
from .simple_matcher import SimpleFaceMatcher

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def register_from_webcam(
    person_id: str,
    max_samples: int = 10,
    min_face_size: int = 80,
):

    # 1. 모델 준비

    face_detector = MediaPipeFaceDetector()
    embedder = ArcFaceONNXEmbedder(
        model_path="arcface_r100.onnx",
        input_size=(112, 112),
        normalize=True,
    )
    matcher = SimpleFaceMatcher()  # 기본 embeddings.json 사용

    # 2. 웹캠 오픈
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("[REGISTER] Cannot open webcam (index 0)")
        return

    log.info("[REGISTER] person_id=%s, max_samples=%d", person_id, max_samples)
    print("")
    print("===================================================")
    print(f"[등록 모드] 사람 ID: {person_id}")
    print("  - 'c' 키: 현재 프레임에서 얼굴 1장 캡처")
    print("  - 'q' 키: 종료 및 저장")
    print(f"  ⇒ 10장을 캡처합니다.")
    print("===================================================")


    collected: List[np.ndarray] = []
    # 추천 10장 캡처 플랜 (각도/포즈 안내)
    pose_plan = [
        "FRONT 1/3",
        "FRONT 2/3 (close your eyes)",
        "FRONT 3/3 (from distance)",
        "LEFT deg 1/2: 15 deg",
        "LEFT deg 2/2: 30 deg",
        "RIGHT deg 1/2: 15 deg",
        "RIGHT deg 2/2: 30 deg",
        "CHIN UP (slightly)",
        "CHIN DOWN (slightly)",
        "EXPR (smile)",
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            log.error("[REGISTER] Failed to read frame from webcam")
            break

        h, w = frame.shape[:2]

        next_pose = "up to you..."

        if len(collected) < 10:
            pose_plan[len(collected)]
        elif len(collected) > max_samples:
            "done! press q"
            
        # 안내 텍스트
        cv2.putText(
            frame,
            f"ID: {person_id} | collected: {len(collected)}/{max_samples}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Press 'c' to capture, 'q' to quit",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Now: {next_pose}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1) & 0xFF

        # 종료
        if key == ord("q"):
            break

        # 캡처
        if key == ord("c"):
            if len(collected) >= max_samples:
                log.info("[REGISTER] Already collected %d samples", max_samples)
                continue

            # 현재 프레임에서 얼굴 탐지
            faces = face_detector.detect_faces(frame)
            if not faces:
                log.info("[REGISTER] No face detected. Try again.")
                continue

            # 가장 큰 얼굴 선택
            f = face_detector.select_main_face(faces)
            x1, y1, x2, y2 = f["box"]
            w_box, h_box = x2 - x1, y2 - y1

            if w_box < min_face_size or h_box < min_face_size:
                log.info(
                    "[REGISTER] Face too small (%dx%d). Move closer.",
                    w_box,
                    h_box,
                )
                continue

            # 얼굴 박스 그리기 (피드백용)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Register Face", frame)
            cv2.waitKey(300)

            # 얼굴 crop
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            face_chip = frame[y1:y2, x1:x2]

            if face_chip.size == 0:
                log.info("[REGISTER] Empty face crop. Skip.")
                continue

            # 임베딩 추출
            try:
                emb = embedder.embed(face_chip)
            except Exception as e:
                log.exception("[REGISTER] embed error: %s", e)
                continue

            collected.append(emb)
            log.info(
                "[REGISTER] Captured %d/%d (face size=%dx%d)",
                len(collected),
                max_samples,
                w_box,
                h_box,
            )

            if len(collected) >= max_samples:
                log.info("[REGISTER] Reached max_samples=%d", max_samples)

    cap.release()
    cv2.destroyAllWindows()

    # 3. 실제 DB에 등록
    if collected:
        matcher.register(person_id, collected)
        log.info("[REGISTER] Done. person_id=%s, samples=%d", person_id, len(collected))
    else:
        log.info("[REGISTER] No samples collected. Nothing saved.")
    

def main():
    parser = argparse.ArgumentParser(description="Register face embeddings from webcam")
    parser.add_argument(
        "--id",
        "--person-id",
        dest="person_id",
        required=True,
        help="등록할 사람 ID (예: Seohyun, Dad 등)",
    )
    parser.add_argument(
        "--n",
        "--trial-n",
        dest="n",
        required=True,
        help="등록할 사람 ID (예: Seohyun, Dad 등)",
    )


    args = parser.parse_args()

    register_from_webcam(
        person_id=args.person_id,
        max_samples=int(args.n),
    )


if __name__ == "__main__":
    main()
