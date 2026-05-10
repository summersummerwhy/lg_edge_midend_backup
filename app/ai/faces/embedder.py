from typing import Tuple
import numpy as np
import cv2
import onnxruntime as ort


class ArcFaceONNXEmbedder():
    """
    ArcFace / InsightFace 계열 ONNX 얼굴 임베딩 모델 래퍼.

    - 입력: 얼굴 crop (BGR, HWC)
    - 출력: 1D embedding (L2-normalized, shape = (D,))

    data_format:
        - "NCHW": (1, C, H, W)  형태로 넣는 모델 (일반적인 insightface 스타일)
        - "NHWC": (1, H, W, C)  형태로 넣는 모델 (garavv/arcface-onnx 등)
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (112, 112),
        data_format: str = "NCHW",
        pixel_mean: float = 0.5,
        pixel_std: float = 0.5,
        scale_01: bool = True,
        normalize: bool = True,
    ):
        """
        Args:
            model_path: .onnx 파일 경로
            input_size: (width, height)
            data_format: "NCHW" or "NHWC"
            pixel_mean, pixel_std:
                - scale_01=True일 때: [-1,1]로 만들고 싶으면 mean=0.5, std=0.5
                - scale_01=False일 때: (x - mean) / std 로 직접 정의
            scale_01:
                - True  : 먼저 /255.0 해서 [0,1]로 만든 뒤 (x - mean)/std
                - False : 바로 (x - mean)/std
            normalize:
                - True면 출력 embedding을 L2 normalize
        """
        self.model_path = model_path
        self.input_size = input_size
        self.data_format = data_format.upper()
        self.scale_01 = scale_01
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.normalize_output = normalize

        if self.data_format not in ("NCHW", "NHWC"):
            raise ValueError(f"Unsupported data_format: {self.data_format}")

        # ONNX Runtime 세션 로드
        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        공통 전처리:
        - BGR -> RGB
        - resize to input_size
        - 정규화 (scale_01, mean/std 조합)
        - data_format에 따라 NCHW 또는 NHWC로 reshape
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Empty face image for embedding")

        img = cv2.resize(face_image, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)

        if self.scale_01:
            img = img / 255.0
            img = (img - self.pixel_mean) / self.pixel_std
        else:
            img = (img - self.pixel_mean) / self.pixel_std

        if self.data_format == "NCHW":
            # HWC -> CHW
            img = np.transpose(img, (2, 0, 1))
            # (1, C, H, W)
            img = np.expand_dims(img, axis=0)
        else:  # "NHWC"
            # 이미 HWC, (1, H, W, C)만 만들어주면 됨
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
