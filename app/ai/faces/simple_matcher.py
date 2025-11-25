import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .base import BaseFaceMatcher, cosine_similarity


class SimpleFaceMatcher(BaseFaceMatcher):
    """
    JSON 기반 Face Matcher

    - embeddings.json 형식:
      {
        "person_id_1": [[...embedding1...], [...embedding2...], ...],
        "person_id_2": [[...], ...],
        ...
      }

    - match(embedding):
      모든 등록 embedding과 cosine similarity를 계산해서
      최대값이 threshold 이상이면 그 사람으로 인식,
      아니면 "unknown" 반환.
    """

    def __init__(self, db_path: str, threshold: float = 0.55):
        self.db_path = Path(db_path)
        self.threshold = threshold
        self._db: Dict[str, List[np.ndarray]] = {}

        # DB 디렉토리 없으면 생성
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._load()

    # ------------------------
    # 내부: JSON <-> 메모리 변환
    # ------------------------

    def _load(self):
        if not self.db_path.exists():
            self._db = {}
            return

        try:
            with self.db_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)

            db: Dict[str, List[np.ndarray]] = {}
            for person_id, emb_list in raw.items():
                db[person_id] = [np.array(e, dtype=np.float32) for e in emb_list]
            self._db = db
        except Exception as e:
            print(f"[FaceMatcher] Failed to load DB: {e}")
            self._db = {}

    def _save(self):
        raw: Dict[str, List[List[float]]] = {}
        for person_id, emb_list in self._db.items():
            raw[person_id] = [emb.astype(float).tolist() for emb in emb_list]

        try:
            with self.db_path.open("w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[FaceMatcher] Failed to save DB: {e}")

    # ------------------------
    # BaseFaceMatcher 인터페이스 구현
    # ------------------------

    def match(self, embedding: np.ndarray) -> Dict:
        """
        Args:
            embedding: L2-normalized 1D vector

        Returns:
            {
                "id": "person_id" or "unknown",
                "score": float,
                "meta": {...}
            }
        """
        if embedding is None or embedding.size == 0:
            return {"id": "unknown", "score": 0.0, "meta": {}}

        best_id = "unknown"
        best_score = -1.0
        best_meta = {}

        for person_id, emb_list in self._db.items():
            for idx, ref_emb in enumerate(emb_list):
                s = cosine_similarity(embedding, ref_emb)
                if s > best_score:
                    best_score = s
                    best_id = person_id
                    best_meta = {"index": idx}

        if best_score < self.threshold:
            return {
                "id": "unknown",
                "score": float(best_score),
                "meta": best_meta,
            }

        return {
            "id": best_id,
            "score": float(best_score),
            "meta": best_meta,
        }

    def register(self, person_id: str, embeddings: List[np.ndarray]):
        """
        Args:
            person_id: "dad", "mom" 같은 식별자
            embeddings: 이 사람의 얼굴 embedding 리스트
        """
        if person_id not in self._db:
            self._db[person_id] = []

        for emb in embeddings:
            if emb is None or emb.size == 0:
                continue
            # L2 normalize 보장
            norm = np.linalg.norm(emb) + 1e-8
            emb_norm = emb / norm
            self._db[person_id].append(emb_norm.astype(np.float32))

        self._save()

    def clear(self):
        """
        DB 초기화
        """
        self._db = {}
        if self.db_path.exists():
            try:
                self.db_path.unlink()
            except Exception as e:
                print(f"[FaceMatcher] Failed to remove DB file: {e}")
