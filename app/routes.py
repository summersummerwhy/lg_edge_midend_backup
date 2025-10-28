from typing import Optional
from fastapi import APIRouter, HTTPException

from .config import MQTT_HOST, MQTT_PORT, AUDIO_DIR, IMAGE_DIR
from .utils import safe_filename
from .state import latest

router = APIRouter()


@router.get("/health")
async def health():
    return {"ok": True, "mqtt": {"host": MQTT_HOST, "port": MQTT_PORT}}


@router.get("/latest/{device}")
async def get_latest(device: str):
    out = {}
    for k in ("motion", "audio", "camera"):
        obj = latest.get((device, k))
        if obj:
            out[k] = obj
    if not out:
        raise HTTPException(404, "No data for device")
    return out


@router.get("/files")
async def list_files(device: Optional[str] = None, kind: Optional[str] = None):
    res = []
    if kind in (None, "audio"):
        base = AUDIO_DIR / (safe_filename(device) if device else "")
        if base.exists():
            for p in base.rglob("*.wav"):
                res.append({"kind": "audio", "path": str(p)})
    if kind in (None, "camera", "image", "images"):
        base = IMAGE_DIR / (safe_filename(device) if device else "")
        if base.exists():
            for p in base.rglob("*.jpg"):
                res.append({"kind": "camera", "path": str(p)})
    return {"count": len(res), "items": res}
