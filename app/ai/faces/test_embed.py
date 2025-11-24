import cv2

from .mp_detector import MediaPipeFaceDetector
from .af_embedder import ArcFaceONNXEmbedder
from .utils import face_box_area, filter_faces_by_area


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    detector = MediaPipeFaceDetector(
        min_confidence=0.5,
        model_selection=0,
    )

    # TODO: 실제 ArcFace ONNX 모델 경로로 바꿔야 함
    embedder = ArcFaceONNXEmbedder(
        model_path="models/arcface.onnx",
        input_size=(112, 112),
        normalize=True,
    )

    MIN_FACE_AREA = 80 * 80  # 80x80 이상만 embed

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        faces = detector.detect_faces(frame)
        big_faces = filter_faces_by_area(faces, min_area=MIN_FACE_AREA)

        for face in faces:
            x1, y1, x2, y2 = face["box"]
            conf = face["confidence"]
            area = face_box_area(face)

            is_big = face in big_faces
            color = (0, 255, 0) if is_big else (0, 0, 255)

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

        # big face가 하나라도 있으면 첫 번째 얼굴 embed 해보기
        if big_faces:
            f = big_faces[0]
            x1, y1, x2, y2 = f["box"]
            face_crop = frame[y1:y2, x1:x2]

            emb = embedder.embed(face_crop)
            # 간단히 shape / norm만 확인
            norm = (emb**2).sum() ** 0.5
            cv2.putText(
                frame,
                f"emb_dim={emb.shape[0]}, norm={norm:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Embedding test (q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
