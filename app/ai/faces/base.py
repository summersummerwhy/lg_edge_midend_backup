"""
Base Face 클래스
모든 얼굴 인식 모델이 상속받는 인터페이스

- BaseFaceDetector : 이미지에서 얼굴 위치 검출
- BaseFaceEmbedder : 얼굴 crop -> embedding 벡터
- BaseFaceMatcher  : embedding -> 사람 ID 매칭
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np


class BaseFaceDetector(ABC):
    """
    얼굴 위치를 찾는 인터페이스.

    input  : 전체 이미지 (BGR, HWC)
    output : 얼굴 박스 목록
        [
            {
                "box": [x1, y1, x2, y2],  # 좌상단 / 우하단
                "confidence": 0.95,
            },
            ...
        ]
    """

    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        pass

    def select_main_face(self, faces: List[Dict]) -> Optional[Dict]:
        """
        여러 얼굴 중 '하나'만 고르고 싶을 때 사용하는 helper.
        기본 구현은 '가장 큰 얼굴'을 고름.
        """
        if not faces:
            return None

        def area(face):
            x1, y1, x2, y2 = face["box"]
            return (x2 - x1) * (y2 - y1)

        return max(faces, key=area)


class BaseFaceEmbedder(ABC):
    """
    얼굴 이미지를 고정 길이 벡터(embedding)로 바꾸는 인터페이스.
    """

    @abstractmethod
    def embed(self, face_image: np.ndarray) -> np.ndarray:
        """
        Args:
            face_image: 얼굴 crop (BGR, HWC)

        Returns:
            1D numpy array, shape = (D,)
        """
        pass

    def embed_batch(self, faces: List[np.ndarray]) -> List[np.ndarray]:
        """
        optional: 배치 embedding이 필요할 때 override 가능.
        기본 구현은 하나씩 embed 호출.
        """
        return [self.embed(f) for f in faces]


class BaseFaceMatcher(ABC):
    """
    embedding 벡터를 기반으로 '누구인지' 판별하는 인터페이스.
    """

    @abstractmethod
    def match(self, embedding: np.ndarray) -> Dict:
        """
        Args:
            embedding: L2-normalized 1D vector

        Returns:
            {
                "id": "person_id" or "unknown",
                "score": float,  # 매칭 점수 (cosine similarity 등)
                "meta": {...}    # optional 부가 정보
            }
        """
        pass

    @abstractmethod
    def register(self, person_id: str, embeddings: List[np.ndarray]):
        """
        얼굴을 등록(Enrollment)할 때 사용하는 메서드.

        Args:
            person_id: "dad", "mom" 같은 식별자
            embeddings: 이 사람의 얼굴 embedding 리스트
        """
        pass

    @abstractmethod
    def clear(self):
        """
        DB 초기화
        """
        pass
