"""
디렉토리에 있는 얼굴 크롭 이미지들로 embeddings.json에 등록하는 CLI

- 입력: 폴더(재귀 가능) 안의 이미지들(.png/.jpg/.jpeg/.bmp/.webp/.ppm/.pgm/.pbm)
- 처리: (선택) 얼굴 재탐지 → 가장 큰 얼굴 crop → ArcFace embed → DB 저장
- 출력: embeddings.json에 person_id 키로 임베딩 리스트 추가/저장

사용 예:
  python -m app.ai.faces.register_face_dir --id Alice --dir /path/to/crops --n 10
  python -m app.ai.faces.register_face_dir --id Seohyun --dir ./face_crops --skip-detect
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detector import MediaPipeFaceDetector
from .embedder import ArcFaceONNXEmbedder
from .simple_matcher import SimpleFaceMatcher

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# OpenCV가 기본 지원하는 NetPBM 계열 포함
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".ppm", ".pgm", ".pbm"}


def _list_images(img_dir: Path, recursive: bool) -> List[Path]:
    if not img_dir.exists() or not img_dir.is_dir():
        raise FileNotFoundError(f"dir not found: {img_dir}")
    if recursive:
        files = [p for p in img_dir.rglob("*") if p.is_file()]
    else:
        files = [p for p in img_dir.iterdir() if p.is_file()]
    files = [p for p in files if p.suffix.lower() in SUPPORTED_EXTS]
    files.sort(key=lambda p: p.as_posix())
    return files


def _read_image(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        log.warning("[READ] failed: %s", path)
    return img


def _clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    return x1, y1, x2, y2


def register_from_directory(
    person_id: str,
    img_dir: Path,
    max_samples: int = 10,
    min_face_size: int = 80,
    recursive: bool = True,
    skip_detect: bool = False,
    model_path: str = "arcface_r100.onnx",
):
    """
    Args:
        person_id: DB에 저장할 사람 ID
        img_dir: 얼굴 크롭 이미지가 들어있는 디렉토리
        max_samples: 최대 몇 장 등록할지 (많으면 파일 앞에서부터 자름)
        min_face_size: (detect 모드일 때) 재탐지된 얼굴 최소 크기
        recursive: 하위폴더까지 탐색
        skip_detect: True면 "이미 얼굴만 들어있는 크롭"이라고 가정하고 얼굴 재탐지 없이 바로 embed
    """
    # 1) 모델 준비
    face_detector = MediaPipeFaceDetector()
    embedder = ArcFaceONNXEmbedder(
        model_path=model_path,
        input_size=(112, 112),
        normalize=True,
    )
    matcher = SimpleFaceMatcher()

    # 2) 파일 수집
    files = _list_images(img_dir, recursive=recursive)
    if not files:
        log.error("[DIR] no images found in %s", img_dir)
        return

    log.info("[DIR] found %d images in %s", len(files), img_dir)
    collected: List[np.ndarray] = []

    # 3) 이미지 → 임베딩
    for path in files:
        if len(collected) >= max_samples:
            break

        img = _read_image(path)
        if img is None:
            continue

        h, w = img.shape[:2]

        face_chip = img

        if not skip_detect:
            faces = face_detector.detect_faces(img)
            if not faces:
                log.info("[SKIP] no face detected: %s", path.name)
                continue

            f = face_detector.select_main_face(faces)
            x1, y1, x2, y2 = f["box"]
            x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, w, h)

            fw, fh = x2 - x1, y2 - y1
            if fw < min_face_size or fh < min_face_size:
                log.info("[SKIP] face too small (%dx%d): %s", fw, fh, path.name)
                continue

            face_chip = img[y1:y2, x1:x2]
            if face_chip.size == 0:
                log.info("[SKIP] empty crop: %s", path.name)
                continue

        # 임베딩
        try:
            emb = embedder.embed(face_chip)
        except Exception as e:
            log.exception("[EMB] embed error (%s): %s", path.name, e)
            continue

        collected.append(emb)
        log.info("[OK] %s (%d/%d)", path.name, len(collected), max_samples)

    # 4) DB 저장
    if collected:
        matcher.register(person_id, collected)
        log.info("[DONE] person_id=%s samples=%d saved_to=%s", person_id, len(collected), matcher.db_path)
    else:
        log.warning("[DONE] no embeddings collected. nothing saved.")


def main():
    parser = argparse.ArgumentParser(description="Register face embeddings from a directory of face crops")
    parser.add_argument("--id", "--person-id", dest="person_id", required=True, help="등록할 사람 ID")
    parser.add_argument("--dir", dest="img_dir", required=True, help="이미지 디렉토리 경로")
    parser.add_argument("--n", dest="n", default="10", help="최대 등록 장수 (default=10)")
    parser.add_argument("--min-face", dest="min_face", default="80", help="최소 얼굴 크기 (detect 모드일 때)")
    parser.add_argument("--no-recursive", action="store_true", help="하위폴더 탐색 안 함")
    parser.add_argument("--skip-detect", action="store_true", help="이미 얼굴 크롭이라고 가정하고 재탐지 스킵")
    parser.add_argument("--model", dest="model_path", default="arcface_r100.onnx", help="ArcFace ONNX 경로")

    args = parser.parse_args()

    register_from_directory(
        person_id=args.person_id,
        img_dir=Path(args.img_dir),
        max_samples=int(args.n),
        min_face_size=int(args.min_face),
        recursive=(not args.no_recursive),
        skip_detect=bool(args.skip_detect),
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
