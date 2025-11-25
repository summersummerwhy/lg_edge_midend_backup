import cv2
import numpy as np
import time

from .detector import MediaPipeFaceDetector
from .embedder import ArcFaceONNXEmbedder
from .simple_matcher import SimpleFaceMatcher


def main():

    face_detector = MediaPipeFaceDetector()
    embedder = ArcFaceONNXEmbedder(
        model_path="arcface_r100.onnx",
        input_size=(112, 112),
        normalize=True,
    )
    matcher = SimpleFaceMatcher()  # embeddings.json 로드됨

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("=== Face Recognize Test ===")
    print("등록된 얼굴이 있으면 ID와 score가 표시됩니다.")
    print("q 키를 누르면 종료합니다.")

    last_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            current_fps = 1.0 / dt
            fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps

        h, w = frame.shape[:2]
        faces = face_detector.detect_faces(frame)

        for f in faces:
            x1, y1, x2, y2 = f["box"]
            face_chip = frame[y1:y2, x1:x2]

            if face_chip.size == 0:
                continue

            emb = embedder.embed(face_chip)
            match = matcher.match(emb)

            face_id = match.get("id", "unknown")
            score = float(match.get("score", 0.0))

            # threshold 예시: 0.6
            if score < 0.6:
                face_id_disp = "unknown"
            else:
                face_id_disp = face_id

            box_w = x2 - x1
            box_h = y2 - y1

            # 박스 & 라벨 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{face_id_disp} ({score:.2f}) [{box_w}x{box_h}]"
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Face Recognize Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
