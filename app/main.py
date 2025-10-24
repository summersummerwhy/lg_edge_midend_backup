import os
import json
import base64
import asyncio
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, PositiveInt
import uvicorn

# === MQTT (asyncio-mqtt) ===
from asyncio_mqtt import Client as MQTTClient, MqttError

from ai.main import track_image, track_image_by_path
from dotenv import load_dotenv


load_dotenv()
# ================== 환경 ==================
MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASS = os.getenv("MQTT_PASS")

MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "topst-receiver")
MQTT_KEEPALIVE = int(os.getenv("MQTT_KEEPALIVE", "60"))
MQTT_QOS = int(os.getenv("MQTT_QOS", "1"))  # 0/1/2

DEVICE_NAMESPACE = os.getenv("DEVICE_NAMESPACE", "topst")  # topst/{device}/...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

DATA_DIR = Path(os.getenv("DATA_DIR", str(DEFAULT_DATA_DIR))).absolute()

AUDIO_DIR = DATA_DIR / "audio"
IMAGE_DIR = DATA_DIR / "images"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

class MotionPayload(BaseModel):
    motion: int = Field(ge=0, le=1)

class AudioPayload(BaseModel):
    format: str = Field(pattern="^(?i)PCM$")
    sample_rate: PositiveInt = 16000
    channel: PositiveInt = 1
    dur_ms: PositiveInt = 20
    data_b64: str

class CameraPayload(BaseModel):
    format: str = Field(pattern="^(?i)jpeg$")
    width: PositiveInt
    height: PositiveInt
    data_b64: Optional[str]
    # 청크 전송일 때 메타
    part: Optional[Literal["start","chunk","end"]] = None
    idx: Optional[int] = None
    total: Optional[int] = None

class Envelope(BaseModel):
    device: str
    ts: int  # UTC ms
    seq: int
    payload: Dict[str, Any]

# ================== 캐시/유틸 ==================
latest: Dict[tuple, Dict[str, Any]] = {}
last_seq: Dict[tuple, int] = {}

def ts_to_iso(ts_ms: int) -> str:
    return datetime.utcfromtimestamp(ts_ms / 1000).isoformat() + "Z"

def is_duplicate(device: str, kind: str, seq: int) -> bool:
    key = (device, kind)
    prev = last_seq.get(key)
    if prev is not None and seq <= prev:
        return True
    last_seq[key] = seq
    return False

def save_wav_pcm16_mono(pcm_bytes: bytes, sample_rate: int, out_path: Path) -> None:
    import wave
    print(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

def safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)

async def handle_motion(msg: Envelope):
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
    print(f"[MOTION] {msg.device} seq={msg.seq} motion={p.motion}")

AUDIO_SAVE_INTERVAL = int(os.getenv("AUDIO_SAVE_INTERVAL", 5))  # WAV 파일 저장 주기 (초)

audio_buffer: Dict[str, list[bytes]] = {}   # device별 PCM 버퍼
audio_last_ts: Dict[str, int] = {}          # 마지막 청크 시각 (ms)
audio_start_ts: Dict[str, int] = {}         # 파일 시작 시각 (ms)

async def handle_audio(msg: Envelope):
    """여러 PCM 청크를 누적하여 일정 주기마다 하나의 WAV 파일로 저장"""
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

    elapsed = (msg.ts - audio_start_ts[device]) / 1000  # 초 단위 경과 시간
    if elapsed >= AUDIO_SAVE_INTERVAL:
        # WAV 저장
        merged = b"".join(audio_buffer[device])
        day = datetime.utcfromtimestamp(msg.ts / 1000).strftime("%Y%m%d")
        subdir = AUDIO_DIR / safe_filename(device) / day
        subdir.mkdir(parents=True, exist_ok=True)
        fname = f"{audio_start_ts[device]}_{msg.seq}_chunk{int(elapsed)}s.wav"
        fpath = subdir / fname

        save_wav_pcm16_mono(merged, p.sample_rate, fpath)
        print(f"[AUDIO] {device} -> saved {fpath.name} ({len(merged)}B, {elapsed:.1f}s)")


        latest[(device, "audio")] = {
            "device": device,
            "ts": msg.ts,
            "seq": msg.seq,
            "payload": {
                "format": "PCM",
                "sample_rate": p.sample_rate,
                "channel": p.channel,
                "dur_ms": AUDIO_SAVE_INTERVAL * 1000,
                "file": str(fpath),
            },
            "ts_iso": ts_to_iso(msg.ts),
        }

        # 버퍼 리셋
        audio_buffer[device].clear()
        audio_start_ts[device] = msg.ts

async def handle_camera(msg: Envelope):
    if is_duplicate(msg.device, "camera", msg.seq):
        return
    p = CameraPayload(**msg.payload)
    jpg = base64.b64decode(p.data_b64)

    day = datetime.utcfromtimestamp(msg.ts / 1000).strftime("%Y%m%d")
    subdir = IMAGE_DIR / safe_filename(msg.device) / day
    fpath = subdir / f"{msg.ts}_{msg.seq}.jpg"
    subdir.mkdir(parents=True, exist_ok=True)
    with open(fpath, "wb") as f:
        f.write(jpg)

    latest[(msg.device, "camera")] = {
        "device": msg.device,
        "ts": msg.ts,
        "seq": msg.seq,
        "payload": {
            "format": "jpeg",
            "width": p.width,
            "height": p.height,
            "file": str(fpath),
        },
        "ts_iso": ts_to_iso(msg.ts),
    }
    print(f"[CAMERA] {msg.device} seq={msg.seq} -> {fpath.name} ({len(jpg)}B)")

    payloads = track_image_by_path(fpath)
    # ai_result = infer_image(fpath)
    # print(f"[AI][CAMERA] {msg.device} -> {ai_result}")

async def route_message(topic: str, payload: bytes):
    try:
        print(payload)
        env = Envelope(**json.loads(payload.decode()))
    except Exception as e:
        print(f"[MQTT] invalid envelope: {e}")
        return

    parts = topic.split("/")
    if len(parts) < 3:
        print(f"[MQTT] bad topic: {topic}")
        return
    ns, device_from_topic, kind = parts[0], parts[1], parts[2]
    if ns != DEVICE_NAMESPACE:
        print(f"[MQTT] namespace mismatch: {topic}")
        return
    if kind == "motion":
        await handle_motion(env)
    elif kind == "audio":
        await handle_audio(env)
    elif kind == "camera":

        await handle_camera(env)
    else:
        print(f"[MQTT] unknown kind: {kind}")

app = FastAPI(title="TOPST Receiver (asyncio-mqtt)", version="0.1.0")

@app.get("/health")
async def health():
    return {"ok": True, "mqtt": {"host": MQTT_HOST, "port": MQTT_PORT}}

@app.get("/latest/{device}")
async def get_latest(device: str):
    out = {}
    for k in ("motion", "audio", "camera"):
        obj = latest.get((device, k))
        if obj:
            out[k] = obj
    if not out:
        raise HTTPException(404, "No data for device")
    return out

@app.get("/files")
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

# ================== MQTT 워커 (asyncio-mqtt) ==================
async def mqtt_worker():
    topics = [
        f"{DEVICE_NAMESPACE}/+/motion",
        f"{DEVICE_NAMESPACE}/+/audio",
        f"{DEVICE_NAMESPACE}/+/camera",
    ]

    backoff = 1
    while True:
        try:
            print(f"[MQTT] connect {MQTT_HOST}:{MQTT_PORT} id={MQTT_CLIENT_ID}")
            async with MQTTClient(
                hostname=MQTT_HOST,
                port=MQTT_PORT,
                client_id=MQTT_CLIENT_ID,
                username=MQTT_USER,
                password=MQTT_PASS,
                keepalive=MQTT_KEEPALIVE,
            ) as client:
                # 구독(QoS 명시)
                for t in topics:
                    await client.subscribe((t, MQTT_QOS))
                    print(f"[MQTT] SUB {t} qos={MQTT_QOS}")

                # 메시지 수신
                async with client.unfiltered_messages() as messages:
                    # 와일드카드 구독 후 모든 메시지를 unfiltered_messages()로 받습니다.
                    for t in topics:
                        await client.subscribe((t, MQTT_QOS))
                    print("[MQTT] waiting messages ...")
                    backoff = 1
                    async for message in messages:
                        try:
                            await route_message(message.topic, message.payload)
                        except Exception as e:
                            print(f"[MQTT] route_message error: {e}")

        except asyncio.CancelledError:
            print("[MQTT] cancelled")
            break
        except MqttError as e:
            print(f"[MQTT] MqttError: {e}")
        except Exception as e:
            print(f"[MQTT] worker error: {e}")

        # 지수 백오프 (최대 30s + 지터)
        sleep_s = min(30, backoff) + (os.getpid() % 10) * 0.01
        print(f"[MQTT] reconnect in {sleep_s:.1f}s")
        await asyncio.sleep(sleep_s)
        backoff *= 2

@app.on_event("startup")
async def on_startup():
    app.state.mqtt_task = asyncio.create_task(mqtt_worker())
    print("[STARTUP] MQTT worker launched.")

@app.on_event("shutdown")
async def on_shutdown():
    task: asyncio.Task = app.state.mqtt_task
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    print("[SHUTDOWN] MQTT worker stopped.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
