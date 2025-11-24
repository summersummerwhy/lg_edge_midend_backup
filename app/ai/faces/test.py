import sys
from pathlib import Path

# mid_end를 sys.path에 추가
ROOT = Path(__file__).resolve().parents[3]  # ~/Desktop/mid_end
sys.path.append(str(ROOT))

import cv2
import numpy as np
from app.ai.faces.mp_detector import MediaPipeFaceDetector
from app.ai.faces.utils import filter_faces_by_area, face_box_area


def main():
    cap = cv2.VideoCapture(0)  # 기본 웹캠

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    detector = MediaPipeFaceDetector(min_confidence=0.5, model_selection=0)

    # 얼굴 박스 최소 area (원하는 대로 조절)
    MIN_FACE_AREA = 80 * 80  # 80x80 이상

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        faces = detector.detect_faces(frame)

        # 크기 기준으로 필터링
        big_faces = filter_faces_by_area(faces, min_area=MIN_FACE_AREA)

        # 박스 그리기
        for face in faces:
            x1, y1, x2, y2 = face["box"]
            conf = face["confidence"]
            area = face_box_area(face)

            # 기준 이상이면 초록색, 아니면 빨간색
            color = (0, 255, 0) if area >= MIN_FACE_AREA else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{conf:.2f} / {area}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("FaceDetector test (q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
