"""
EfficientDet-Lite Detector
PyTorch effdet 사용
"""

from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
from effdet import create_model

from .base import BaseDetector


class EfficientDetLiteDetector(BaseDetector):
    """
    EfficientDet Lite 계열 감지 모델 
    """

    def __init__(
        self,
        model_name: str = "tf_efficientdet_lite0",
        confidence: float = 0.5,
        device: Optional[str] = None,
        pretrained: bool = True,
    ):
        super().__init__(confidence=confidence)

        # device 설정
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # effdet에서 predict용 bench 모델 생성
        self.model = create_model(
            model_name,
            bench_task="predict",
            pretrained=pretrained,
        ).to(self.device)
        self.model.eval()

        # 입력 이미지 크기 (ex. (320, 320) for lite0)
        img_size = self.model.config.image_size

        # img_size 타입이 int / float / tuple / list / ListConfig 등
        # 뭐가 오든 안정적으로 처리하도록 분기
        if isinstance(img_size, (int, float)):
            # 예: img_size = 320
            self.input_h = self.input_w = int(img_size)
        else:
            # 예: img_size = [320, 320] 이거나 ListConfig([320, 320]) 등
            try:
                size_list = list(img_size)  # ListConfig -> list 강제 변환
                if len(size_list) == 2:
                    self.input_h, self.input_w = int(size_list[0]), int(size_list[1])
                else:
                    # 혹시 예상 못한 형태면 그냥 첫 값으로 정사각형 처리
                    self.input_h = self.input_w = int(size_list[0])
            except TypeError as e:
                raise TypeError(
                    f"Unexpected image_size type: {type(img_size)} / value={img_size}"
                ) from e


    @torch.inference_mode()
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        이미지에서 사람 감지

        Args:
            image: OpenCV BGR 이미지 (H, W, 3)

        Returns:
            BaseDetector 규격의 detection 리스트
        """
        if image is None or image.size == 0:
            return []

        orig_h, orig_w = image.shape[:2]

        # ---- 전처리: BGR -> RGB, 리사이즈, normalize ----
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (self.input_w, self.input_h))

        # [H,W,3] -> [1,3,H,W], float32, 0~1
        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 원본 ↔ 입력 스케일
        x_scale = orig_w / self.input_w
        y_scale = orig_h / self.input_h

        # ---- 추론 (여기서 kwargs 제거!) ----
        outputs = self.model(tensor)
        detections: List[Dict] = []

        # effdet 버전에 따라 두 가지 케이스를 처리:
        # 1) outputs: list[dict] with keys: "boxes", "scores", "labels"
        # 2) outputs: Tensor / list[Tensor] with shape [N,6] or [1,N,6] (x1,y1,x2,y2,score,cls)
        if isinstance(outputs, (list, tuple)):
            out0 = outputs[0]

            # --- 케이스 1: dict 형식 ({'boxes','scores','labels'}) ---
            if isinstance(out0, dict) and "boxes" in out0:
                boxes = out0["boxes"].detach().cpu().numpy()
                scores = out0["scores"].detach().cpu().numpy()
                labels = out0["labels"].detach().cpu().numpy().astype(int)

                for box, score, cls in zip(boxes, scores, labels):
                    score = float(score)
                    if score < self.confidence:
                        continue

                    x1, y1, x2, y2 = box.tolist()
                    # 모델 입력 기준 좌표 → 원본 좌표로 스케일링
                    x1 *= x_scale
                    x2 *= x_scale
                    y1 *= y_scale
                    y2 *= y_scale

                    detections.append(
                        {
                            "box": [x1, y1, x2, y2],
                            "confidence": score,
                            "class_id": int(cls),
                        }
                    )

            # --- 케이스 2: Tensor 형식 ([N,6] / [1,N,6]) ---
            else:
                if isinstance(out0, torch.Tensor):
                    det_tensor = out0
                else:
                    # 혹시 모를 타입 대비
                    det_tensor = torch.as_tensor(out0)

                arr = det_tensor.detach().cpu().numpy()
                # [1, N, 6] → [N,6]
                if arr.ndim == 3:
                    arr = arr[0]

                for det in arr:
                    if len(det) < 6:
                        continue
                    x1, y1, x2, y2, score, cls = det.tolist()
                    score = float(score)
                    if score < self.confidence:
                        continue

                    x1 *= x_scale
                    x2 *= x_scale
                    y1 *= y_scale
                    y2 *= y_scale

                    detections.append(
                        {
                            "box": [x1, y1, x2, y2],
                            "confidence": score,
                            "class_id": int(cls),
                        }
                    )

        # --- 케이스 3: outputs 자체가 Tensor인 경우 ([N,6]/[1,N,6]) ---
        elif isinstance(outputs, torch.Tensor):
            arr = outputs.detach().cpu().numpy()
            if arr.ndim == 3:
                arr = arr[0]

            for det in arr:
                if len(det) < 6:
                    continue
                x1, y1, x2, y2, score, cls = det.tolist()
                score = float(score)
                if score < self.confidence:
                    continue

                x1 *= x_scale
                x2 *= x_scale
                y1 *= y_scale
                y2 *= y_scale

                detections.append(
                    {
                        "box": [x1, y1, x2, y2],
                        "confidence": score,
                        "class_id": int(cls),
                    }
                )

        # 사람(class_id=0)만 필터링
        return self.filter_persons(detections)


    def warmup(self):
        """
        더미 이미지로 워밍업 (첫 추론 속도 개선)
        """
        dummy = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
        _ = self.detect(dummy)
        print("[EfficientDetLite] Warmup complete")
