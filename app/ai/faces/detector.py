from typing import List, Dict
import cv2
import numpy as np
import mediapipe as mp


class MediaPipeFaceDetector():
    """
    MediaPipe Face Detection 기반 FaceDetector

    - 입력: BGR 이미지 (H, W, 3)
    - 출력: BaseFaceDetector 인터페이스에 맞는 faces 리스트
      [
        {
          "box": [x1, y1, x2, y2],
          "confidence": float,
        },
        ...
      ]
    """

    def __init__(self, min_confidence: float = 0.5, model_selection: int = 0):
        """
        Args:
            min_confidence: 최소 confidence threshold
            model_selection:
                0: 0.5m 이내 얼굴 (웹캠, 셀카 위주)
                1: 좀 더 먼 거리
        """
        self.min_confidence = min_confidence
        self.model_selection = model_selection

        self._mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=self.model_selection,
            min_detection_confidence=self.min_confidence,
        )

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        h, w, _ = image.shape

        # MediaPipe는 RGB 입력
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._mp_face.process(rgb)

        faces: List[Dict] = []
        if not results.detections:
            return faces

        for det in results.detections:
            score = det.score[0]
            if score < self.min_confidence:
                continue

            bbox = det.location_data.relative_bounding_box

            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            # 이미지 범위로 클리핑
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            # 잘못된 박스 제거
            if x2 <= x1 or y2 <= y1:
                continue

            faces.append({
                "box": [x1, y1, x2, y2],
                "confidence": float(score),
            })

        return faces
