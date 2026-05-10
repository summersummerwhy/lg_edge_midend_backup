# download_face_models.py
from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path

"""
지원 모델 목록
- repo_id: Hugging Face 모델 저장소
- filename: 실제 다운로드할 파일명
- local_name: 로컬에 저장할 파일명
"""

MODEL_SOURCES = [
    {
        "name": "arcface_r100",
        "repo_id": "FoivosPar/Arc2Face",
        "filename": "arcface.onnx",
        "local_name": "arcface_r100.onnx",
    },
    {
        "name": "mobilefacenet",
        "repo_id": "xuexingyu24/insightface-facerecognition",
        "filename": "mobilefacenet.onnx",
        "local_name": "mobilefacenet.onnx",
    },
    {
        "name": "arcface_r50",
        "repo_id": "wujiyang/Face_PyTorch",
        "filename": "w600k_r50.onnx",
        "local_name": "arcface_r50.onnx",
    },
]


def download_models():
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "app" / "ai" / "faces" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Model target directory: {model_dir}")

    for model in MODEL_SOURCES:
        name = model["name"]
        repo_id = model["repo_id"]
        filename = model["filename"]
        local_name = model["local_name"]

        print(f"\n[CHECK] {name} ({filename}) in {repo_id} ...")

        try:
            # 1) repo 파일 목록 조회
            files = list_repo_files(repo_id)
        except Exception as e:
            print(f"[SKIP] Cannot access repo '{repo_id}': {e}")
            continue

        # 2) 파일 존재 여부 확인
        if filename not in files:
            print(f"[SKIP] File '{filename}' not found in repo '{repo_id}'")
            continue

        print(f"[OK] File exists. Downloading...")

        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
            )

            # 다운로드 후 local_name으로 rename
            target_path = model_dir / local_name
            Path(local_path).rename(target_path)

            print(f"[DONE] Saved as: {target_path}")

        except Exception as e:
            print(f"[ERROR] Failed to download {name}: {e}")


if __name__ == "__main__":
    download_models()
