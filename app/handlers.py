import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import cv2

from app.ai.main import track_image_by_path, track_image
from app.ai.temp import add_new_face
from .config import AUDIO_DIR, IMAGE_DIR, DEVICE_NAMESPACE, AUDIO_SAVE_INTERVAL
from .models import Envelope, MotionPayload, AudioPayload, CameraPayload, FacePayload
from .utils import ts_to_iso, safe_filename, save_wav_pcm16_mono
from .state import (
    latest,
    is_duplicate,
    audio_buffer,
    audio_last_ts,
    audio_start_ts,
    camera_chunks,
    ChunkState,
)
from .mqtt.publisher import publish_mqtt

log = logging.getLogger(__name__)
raw_camera_headers: Dict[Tuple[str, int], Dict[str, Any]] = {}
raw_camera_chunks: Dict[Tuple[str, int], Dict[int, bytes]] = {}


def _save_image_file(device: str, ts: int, seq: int, jpg: bytes) -> Path:
    day = datetime.utcfromtimestamp(ts / 1000).strftime("%Y%m%d")
    subdir = IMAGE_DIR / safe_filename(device) / day
    subdir.mkdir(parents=True, exist_ok=True)
    fpath = subdir / f"{ts}_{seq}.jpg"
    with open(fpath, "wb") as f:
        f.write(jpg)
    return fpath


def face_ai(img: np.ndarray) -> None:
    """
    가상의 Face AI 함수
    """
    log.info(f"[FACE_AI] Processing image shape={img.shape}")


async def handle_motion(msg: Envelope) -> None:
    if is_duplicate(msg.device, "motion", msg.seq):
        return
    p = MotionPayload(**msg.payload)
    latest[(msg.device, "motion")] = {
        "device": msg.device,
        "ts": msg.ts,
        "seq": msg.seq,
        "payload": {"motion": p.motion},
        "ts_iso": ts_to_iso(msg.ts),
    }
    log.info("[MOTION] %s seq=%s motion=%s", msg.device, msg.seq, p.motion)


async def handle_audio(msg: Envelope) -> None:
    if is_duplicate(msg.device, "audio", msg.seq):
        return

    p = AudioPayload(**msg.payload)
    pcm = base64.b64decode(p.data_b64)
    device = msg.device

    if device not in audio_buffer:
        audio_buffer[device] = []
        audio_start_ts[device] = msg.ts
        audio_last_ts[device] = msg.ts

    audio_buffer[device].append(pcm)
    audio_last_ts[device] = msg.ts

    elapsed = (msg.ts - audio_start_ts[device]) / 1000
    if elapsed >= AUDIO_SAVE_INTERVAL:
        merged = b"".join(audio_buffer[device])
        day = datetime.utcfromtimestamp(msg.ts / 1000).strftime("%Y%m%d")
        subdir = AUDIO_DIR / safe_filename(device) / day
        subdir.mkdir(parents=True, exist_ok=True)
        fname = f"{audio_start_ts[device]}_{msg.seq}_chunk{int(elapsed)}s.wav"
        fpath = subdir / fname

        save_wav_pcm16_mono(merged, p.sample_rate, fpath)
        log.info(
            "[AUDIO] %s -> saved %s (%dB, %.1fs)",
            device,
            fpath.name,
            len(merged),
            elapsed,
        )

        latest[(device, "audio")] = {
            "device": device,
            "ts": msg.ts,
            "seq": msg.seq,
            "payload": {
                "format": "PCM",
                "sample_rate": p.sample_rate,
                "channel": p.channel,
                "dur_ms": int(elapsed * 1000),
                "file": str(fpath),
            },
            "ts_iso": ts_to_iso(msg.ts),
        }

        audio_buffer[device].clear()
        audio_start_ts[device] = msg.ts


async def handle_camera(msg: Envelope, publish_cb, *, save_image: bool = True) -> None:
    """
    save_image=True  → JPEG 파일 저장 후 track_image_by_path() 사용
    save_image=False → JPEG 메모리 디코드 후 track_image() 사용 (파일 미저장)
    """

    # if is_duplicate(msg.device, "camera", msg.seq):
    #     return

    p = CameraPayload(**msg.payload)
    width, height = p.width, p.height
    jpg_bytes: Optional[bytes] = None
    fpath: Optional[Path] = None

    # (1) 단일 전송
    if p.part is None:
        if not p.data_b64:
            log.warning("[CAMERA] %s seq=%s no data_b64", msg.device, msg.seq)
            return
        jpg_bytes = base64.b64decode(p.data_b64)
        if save_image:
            fpath = _save_image_file(msg.device, msg.ts, msg.seq, jpg_bytes)
            log.info("[CAMERA] %s seq=%s -> %s", msg.device, msg.seq, fpath.name)

    # (2) 청크 전송
    else:
        key: Tuple[str, int] = (msg.device, msg.seq)
        st = camera_chunks.get(key)
        if st is None:
            st = ChunkState(start_ts=msg.ts)
            camera_chunks[key] = st

        if p.total:
            st.set_total(p.total)
        if p.idx is None:
            log.warning("[CAMERA] %s seq=%s missing idx", msg.device, msg.seq)
            return
        if p.data_b64:
            st.add(p.idx, p.data_b64)

        if (p.part == "end") or st.complete():
            missing = []
            if st.total:
                have = set(st.received)
                missing = [i for i in range(st.total) if i not in have]

            if missing:
                ordered = [st.parts[i] for i in sorted(st.received)]
                jpg_bytes = base64.b64decode("".join(ordered))
                if save_image:
                    fpath = _save_image_file(
                        msg.device, st.start_ts, msg.seq, jpg_bytes
                    )
                log.warning(
                    "[CAMERA] %s seq=%s partial (%dB, %d/%s) missing=%s",
                    msg.device,
                    msg.seq,
                    len(jpg_bytes),
                    len(st.received),
                    st.total,
                    missing[:5],
                )
            else:
                if st.total is None:
                    ordered = [st.parts[i] for i in sorted(st.received)]
                else:
                    ordered = [st.parts[i] for i in range(st.total)]
                jpg_bytes = base64.b64decode("".join(ordered))
                if save_image:
                    fpath = _save_image_file(
                        msg.device, st.start_ts, msg.seq, jpg_bytes
                    )
                log.info(
                    "[CAMERA] %s seq=%s full (%dB, %d chunks)",
                    msg.device,
                    msg.seq,
                    len(jpg_bytes),
                    len(ordered),
                )
            camera_chunks.pop(key, None)
        else:
            return

    # latest 캐시
    latest_payload = {"format": "jpeg", "width": width, "height": height}
    if fpath is not None:
        latest_payload["file"] = str(fpath)

    latest[(msg.device, "camera")] = {
        "device": msg.device,
        "ts": msg.ts,
        "seq": msg.seq,
        "payload": latest_payload,
        "ts_iso": ts_to_iso(msg.ts),
    }

    # AI 호출
    try:
        if save_image:
            payloads = track_image_by_path(fpath)
        else:
            if jpg_bytes is None:
                log.warning(
                    "[AI][CAMERA] %s seq=%s no jpg bytes to process",
                    msg.device,
                    msg.seq,
                )
                return
            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                log.error(
                    "[AI][CAMERA] %s seq=%s failed to decode JPEG", msg.device, msg.seq
                )
                return
            payloads = track_image(image, "jpg")

        if not payloads:
            log.info("[AI][CAMERA] %s -> no objects", msg.device)
            return

        for payload in payloads:
            ai_msg = {
                "device": msg.device,
                "ts": msg.ts,
                "seq": msg.seq,
                "payload": payload,
            }
            topic = f"{DEVICE_NAMESPACE}/{DEVICE_NAMESPACE}/ai"
            await publish_cb(topic, ai_msg)
            log.debug(
                "[AI-PUB] %s (%s, track_id=%s)",
                topic,
                payload.get("type"),
                payload.get("track_id"),
            )
    except Exception as e:
        log.exception("[AI][CAMERA] error: %s", e)


async def handle_camera_header_raw(env: Envelope) -> None:
    key = (env.device, env.seq)
    raw_camera_headers[key] = {
        "ts": env.ts,
        **env.payload,
    }
    await _try_assemble_raw(env.device, env.seq)


async def handle_camera_chunk_raw(
    *,
    device: str,
    seq: int,
    idx: int,
    chunk_bytes: bytes,
    save_image: bool = True,
) -> None:
    key = (device, seq)
    if key not in raw_camera_chunks:
        raw_camera_chunks[key] = {}
    if idx not in raw_camera_chunks[key]:
        raw_camera_chunks[key][idx] = chunk_bytes

    await _try_assemble_raw(device, seq, save_image=save_image)


async def handle_face(msg: Envelope) -> None:
    p = FacePayload(**msg.payload)

    try:
        img_bytes = base64.b64decode(p.data)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            log.error("[FACE] %s seq=%s failed to decode image", msg.device, msg.seq)
            return

        log.info(
            "[FACE] %s seq=%s received face image %dx%d",
            msg.device,
            msg.seq,
            p.width,
            p.height,
        )

        add_new_face(img)

    except Exception as e:
        log.exception("[FACE] error: %s", e)


async def _try_assemble_raw(device: str, seq: int, *, save_image: bool = True) -> None:
    key = (device, seq)
    header = raw_camera_headers.get(key)
    chunks = raw_camera_chunks.get(key)
    if not header or not chunks:
        return

    total_chunks = header.get("chunks")
    total_size = header.get("size")
    width = header.get("width")
    height = header.get("height")
    ts = header.get("ts")

    if total_chunks is None or len(chunks) < total_chunks:
        return

    parts = []
    for i in range(total_chunks):
        part = chunks.get(i)
        if part is None:
            return
        parts.append(part)
    jpg_bytes = b"".join(parts)

    fpath = None
    if save_image:
        fpath = _save_image_file(device, ts, seq, jpg_bytes)
        log.info(
            "[CAMERA-RAW] %s seq=%s saved %s (%dB, %d chunks)",
            device,
            seq,
            fpath.name,
            len(jpg_bytes),
            total_chunks,
        )

    # latest 갱신
    from .state import latest

    latest_payload = {"format": "jpeg", "width": width, "height": height}
    if fpath is not None:
        latest_payload["file"] = str(fpath)
    latest[(device, "camera")] = {
        "device": device,
        "ts": ts,
        "seq": seq,
        "payload": latest_payload,
        "ts_iso": ts_to_iso(ts),
    }

    # AI 호출
    try:
        if save_image and fpath is not None:
            payloads = track_image_by_path(fpath)
        else:
            import numpy as np, cv2

            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                log.error("[AI][CAMERA] %s seq=%s JPEG decode 실패", device, seq)
                return
            payloads = track_image(image, "jpg")

        if not payloads:
            log.info("[AI][CAMERA] %s -> no payloads", device)
            return

        topic = f"{DEVICE_NAMESPACE}/{DEVICE_NAMESPACE}/ai"

        for payload in payloads:
            ai_msg = {
                "device": device,
                "ts": ts,
                "seq": seq,
                "payload": payload,
            }
            await publish_mqtt(topic, ai_msg)
            log.debug(
                "[AI-PUB] %s (%s, track_id=%s)",
                topic,
                payload.get("type"),
                payload.get("track_id"),
            )

    except Exception as e:
        log.exception("[AI][CAMERA] error: %s", e)

    # 메모리 정리
    raw_camera_headers.pop(key, None)
    raw_camera_chunks.pop(key, None)
