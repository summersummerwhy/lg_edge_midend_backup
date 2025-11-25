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

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)