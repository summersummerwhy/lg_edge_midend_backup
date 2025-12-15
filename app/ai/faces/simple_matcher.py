import json
from pathlib import Path
import logging
from typing import Dict, List, Optional

import numpy as np

from .base import BaseFaceMatcher, cosine_similarity

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SimpleFaceMatcher(BaseFaceMatcher):
    """
    - embeddings.json 파일에 사람별 임베딩들을 저장
    - 매칭 시: 모든 사람의 모든 임베딩과 cosine similarity를 계산해서
      가장 높은 것 하나를 반환
    """

    def __init__(self, db_path: Optional[str] = None, threshold: float = 0.54):
        """
        Args:
            db_path: embeddings.json 경로 (default -> faces/database/embeddings.json)
        """
        self.threshold = threshold
        if db_path is None:
            faces_dir = Path(__file__).resolve().parent
            db_dir = faces_dir / "database"
            db_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = db_dir / "embeddings.json"
        else:
            self.db_path = Path(db_path)

        # 내부 메모리 DB
        # { "Seohyun": [np.array(...), np.array(...), ...], ... }
        self._db: Dict[str, List[np.ndarray]] = {}

        self._load_db()

    # --------------------- DB I/O --------------------- #

    def _load_db(self):
        if not self.db_path.exists():
            log.info("[FaceDB] %s not found. starting with empty DB.", self.db_path)
            self._db = {}
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            log.exception("[FaceDB] failed to load %s: %s", self.db_path, e)
            self._db = {}
            return

        db: Dict[str, List[np.ndarray]] = {}
        # raw 형식: { "Alice": [ [float, ...], [float, ...], ... ], ... }
        for person_id, emb_list in raw.items():
            db[person_id] = [np.array(e, dtype=np.float32) for e in emb_list]

        self._db = db
        log.info(
            "[FaceDB] loaded %d persons from %s",
            len(self._db),
            self.db_path,
        )

    def _save_db(self):
        raw: Dict[str, List[List[float]]] = {}
        for person_id, emb_list in self._db.items():
            raw[person_id] = [e.astype(float).tolist() for e in emb_list]

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(raw, f, ensure_ascii=False)
            log.info(
                "[FaceDB] saved %d persons to %s",
                len(self._db),
                self.db_path,
            )
        except Exception as e:
            log.exception("[FaceDB] failed to save %s: %s", self.db_path, e)

    # --------------------- BaseFaceMatcher 구현 --------------------- #

    def match(self, embedding: np.ndarray) -> Dict:
        """
        가장 유사한 사람 한 명을 찾아 반환.
        (개선) 사람별로 similarity를 모아서 top-k 평균으로 판단
        """
        if not self._db:
            return {
                "id": "unknown",
                "score": 0.0,
                "meta": {"reason": "empty_db", "db_size": 0},
            }

        TOPK = 3  # 필요하면 1~5 사이로 조절 (3 추천)

        best_person = None
        best_score = -1.0

        for person_id, emb_list in self._db.items():
            if not emb_list:
                continue

            scores = [cosine_similarity(embedding, ref) for ref in emb_list]
            scores.sort(reverse=True)

            k = min(TOPK, len(scores))
            agg = float(sum(scores[:k]) / k)  # top-k 평균

            if agg > best_score:
                best_score = agg
                best_person = person_id

        if best_person is None or best_score < self.threshold:
            return {
                "id": "unknown",
                "score": 0.0,
                "meta": {"reason": "no_match", "db_size": len(self._db)},
            }

        return {
            "id": best_person,
            "score": float(best_score),
            "meta": {
                "best_person": best_person,
                "best_score": float(best_score),
                "db_size": len(self._db),
                "method": f"top{TOPK}_mean",
            },
        }


    def register(self, person_id: str, embeddings: List[np.ndarray]):
        """
        같은 사람의 얼굴 임베딩 여러 개를 한 번에 등록
        embeddings 안의 벡터는 이미 L2-normalized라고 가정
        """
        if not embeddings:
            log.warning("[FaceDB] register called with empty embeddings for %s", person_id)
            return

        # numpy 배열로 강제 캐스팅
        embs = [np.asarray(e, dtype=np.float32) for e in embeddings]

        if person_id not in self._db:
            self._db[person_id] = embs
        else:
            self._db[person_id].extend(embs)

        log.info(
            "[FaceDB] registered %d embeddings for %s (total %d)",
            len(embs),
            person_id,
            len(self._db[person_id]),
        )

        self._save_db()

    def clear(self):
        """
        DB 초기화 (메모리 + 파일 둘 다)
        """
        self._db.clear()
        try:
            if self.db_path.exists():
                self.db_path.unlink()
                log.info("[FaceDB] %s removed", self.db_path)
        except Exception as e:
            log.exception("[FaceDB] failed to remove %s: %s", self.db_path, e)