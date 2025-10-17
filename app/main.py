import asyncio
import json
import os
import orjson
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from asyncio_mqtt import Client, MqttError
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("topst")

MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER", "server")
MQTT_PASS = os.getenv("MQTT_PASS", "server_pass")

SUB_TOPICS = [
    "topst/+/telemetry",
    "topst/+/motion",
    "topst/+/audio",
    "topst/+/image",
    "topst/#"
]

class Envelope(BaseModel):
    v: int = 1
    type: Literal["telemetry", "motion", "audio", "image"]
    device: str
    ts: int
    seq: int
    payload: dict

class MotionPayload(BaseModel):
    state: Literal["ON", "OFF"]
    dur_ms: Optional[int] = None
    pin: Optional[int] = None
    zone: Optional[str] = None

class AudioFeatPayload(BaseModel):
    mode: Literal["features", "raw_ref"]
    sr: int
    win_ms: Optional[int] = None
    hop_ms: Optional[int] = None
    feat: Optional[dict] = None
    url: Optional[str] = None
    fmt: Optional[Literal["s16le", "adpcm", "mulaw"]] = None
    dur_ms: Optional[int] = None
    sha256: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(mqtt_worker())
    try:
        yield
    finally:
        task.cancel()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"ok": True}

subscribers: set[asyncio.Queue] = set()

async def broadcast(event_type: str, data: dict):
    # 추후 RPi가 /stream/alerts 또는 /stream/raw 를 구독 가능
    msg = {"event": event_type, "data": data}
    for q in list(subscribers):
        try:
            await q.put(msg)
        except RuntimeError:
            subscribers.discard(q)

@app.get("/stream/raw")
async def stream_raw(request: Request):
    q: asyncio.Queue = asyncio.Queue()
    subscribers.add(q)

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"event: {item['event']}\n" \
                          f"data: {orjson.dumps(item['data']).decode()}\n\n"
                except asyncio.TimeoutError:
                    yield "event: heartbeat\ndata: {}\n\n"
        finally:
            subscribers.discard(q)

    return StreamingResponse(gen(), media_type="text/event-stream")

async def mqtt_worker():
    while True:
        try:
            log.info(f"[MQTT] connecting to {MQTT_HOST}:{MQTT_PORT} ...")
            async with Client(
                hostname=MQTT_HOST,
                port=MQTT_PORT,
                # username=MQTT_USER,
                # password=MQTT_PASS,
                client_id="topst-server",
                keepalive=60,
            ) as client:

                async with client.unfiltered_messages() as messages:
                    for t in SUB_TOPICS:
                        await client.subscribe(t)
                    log.info(f"[MQTT] subscribed: {SUB_TOPICS}")

                    async for msg in messages:
                        topic = msg.topic
                        txt = msg.payload.decode(errors="ignore")
                        log.info(f"[MQTT] {topic} {txt}")

                        try:
                            payload = json.loads(txt)
                            env = Envelope(**payload)
                        except Exception as e:
                            log.warning(f"[MQTT] JSON/Pydantic skipped: {e}")
                            continue

                        if env.type == "motion":
                            try:
                                MotionPayload(**env.payload)
                            except Exception as e:
                                log.warning(f"[MQTT] Motion payload invalid: {e}")
                                continue
                            await broadcast("motion", payload)

                        elif env.type == "audio":
                            try:
                                AudioFeatPayload(**env.payload)
                            except Exception as e:
                                log.warning(f"[MQTT] Audio payload invalid: {e}")
                                continue
                            await broadcast("audio", payload)

                        elif env.type == "image":
                            await broadcast("image", payload)

                        elif env.type == "telemetry":
                            await broadcast("telemetry", payload)

        except MqttError as e:
            log.error(f"[MQTT] error: {e!r}")
            await asyncio.sleep(2)
        except Exception as e:
            log.exception(f"[MQTT] worker crashed: {e}")
            await asyncio.sleep(2)
