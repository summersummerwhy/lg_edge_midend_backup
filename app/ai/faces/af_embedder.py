from typing import Tuple
import numpy as np
import cv2
import onnxruntime as ort

from .base import BaseFaceEmbedder


class ArcFaceONNXEmbedder(BaseFaceEmbedder):
    """
    ArcFace / InsightFace 계열 ONNX 얼굴 임베딩 모델 래퍼.

    - 입력: 얼굴 crop (BGR, HWC)
    - 출력: 1D embedding (L2-normalized, shape = (D,))
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (112, 112),
        normalize: bool = True,
    ):
        """
        Args:
            model_path: .onnx 파일 경로
            input_size: (width, height) - 모델에 따라 112x112, 128x128 등
            normalize: 출력 벡터를 L2 normalize 할지 여부
        """
        self.model_path = model_path
        self.input_size = input_size
        self.normalize_output = normalize

        # ONNX Runtime 세션 로드
        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        ArcFace 계열 전처리 (예시):
        - BGR -> RGB
        - resize to input_size
        - [0,1] 스케일 후 [-1,1] 정규화
        - NCHW, batch dimension 추가
        """
        # 얼굴이 너무 작은 경우 방어 코드
        if face_image is None or face_image.size == 0:
            raise ValueError("Empty face image for embedding")

        img = cv2.resize(face_image, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # [0,1] -> [-1,1]

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        # (1, C, H, W)
        img = np.expand_dims(img, axis=0)
        return img

    def embed(self, face_image: np.ndarray) -> np.ndarray:
        inp = self._preprocess(face_image)

        emb = self.session.run(
            [self.output_name],
            {self.input_name: inp},
        )[0]

        emb = emb.squeeze()  # (D,)

        if self.normalize_output:
            norm = np.linalg.norm(emb) + 1e-8
            emb = emb / norm

        return emb.astype(np.float32)

    def embed_batch(self, faces):
        return [self.embed(f) for f in faces]
