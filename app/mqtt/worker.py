import os
import asyncio
import json
from asyncio_mqtt import Client as MQTTClient, MqttError
import logging

from ..config import (
    MQTT_HOST,
    MQTT_PORT,
    MQTT_USER,
    MQTT_PASS,
    MQTT_CLIENT_ID,
    MQTT_KEEPALIVE,
    MQTT_QOS,
    DEVICE_NAMESPACE,
    SAVE_CAMERA_FILES,
)
from ..models import Envelope
from ..handlers import handle_motion, handle_audio, handle_camera
from .publisher import publish_mqtt

log = logging.getLogger(__name__)


async def route_message(topic: str, payload: bytes):
    try:
        env = Envelope(**json.loads(payload.decode()))
    except Exception as e:
        log.warning("[MQTT] invalid envelope: %s", e)
        return

    parts = topic.split("/")
    if len(parts) < 3:
        log.warning("[MQTT] bad topic: %s", topic)
        return

    ns, device_from_topic, kind = parts[0], parts[1], parts[2]
    if ns != DEVICE_NAMESPACE:
        log.debug("[MQTT] namespace mismatch: %s", topic)
        return

    if kind == "motion":
        await handle_motion(env)
    elif kind == "audio":
        await handle_audio(env)
    elif kind == "camera":
        await handle_camera(env, publish_cb=publish_mqtt, save_image=SAVE_CAMERA_FILES)
    else:
        log.debug("[MQTT] unknown kind: %s", kind)


async def mqtt_worker():
    topics = [
        f"{DEVICE_NAMESPACE}/+/motion",
        f"{DEVICE_NAMESPACE}/+/audio",
        f"{DEVICE_NAMESPACE}/+/camera",
    ]

    backoff = 1
    while True:
        try:
            log.info("[MQTT] connect %s:%s id=%s", MQTT_HOST, MQTT_PORT, MQTT_CLIENT_ID)
            async with MQTTClient(
                hostname=MQTT_HOST,
                port=MQTT_PORT,
                client_id=MQTT_CLIENT_ID,
                username=MQTT_USER,
                password=MQTT_PASS,
                keepalive=MQTT_KEEPALIVE,
            ) as client:
                for t in topics:
                    await client.subscribe((t, MQTT_QOS))
                    log.info("[MQTT] SUB %s qos=%s", t, MQTT_QOS)

                async with client.unfiltered_messages() as messages:
                    log.info("[MQTT] waiting messages ...")
                    backoff = 1
                    async for message in messages:
                        try:
                            await route_message(message.topic, message.payload)
                        except Exception as e:
                            log.exception("[MQTT] route_message error: %s", e)

        except asyncio.CancelledError:
            log.info("[MQTT] cancelled")
            break
        except MqttError as e:
            log.warning("[MQTT] MqttError: %s", e)
        except Exception as e:
            log.exception("[MQTT] worker error: %s", e)

        sleep_s = min(30, backoff) + (os.getpid() % 10) * 0.01
        log.info("[MQTT] reconnect in %.1fs", sleep_s)
        await asyncio.sleep(sleep_s)
        backoff *= 2
