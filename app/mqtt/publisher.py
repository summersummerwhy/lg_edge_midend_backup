import json
import asyncio
from typing import Any, Dict, Optional
from asyncio_mqtt import Client as MQTTClient, MqttError
import logging

from ..config import (
    MQTT_HOST,
    MQTT_PORT,
    MQTT_USER,
    MQTT_PASS,
    MQTT_CLIENT_ID,
    MQTT_KEEPALIVE,
)
from ..state import mqtt_lock

log = logging.getLogger(__name__)
_mqtt_client: Optional[MQTTClient] = None
seq_number = 0

async def _connect_client() -> MQTTClient:
    global _mqtt_client
    if _mqtt_client is not None:
        return _mqtt_client
    client = MQTTClient(
        hostname=MQTT_HOST,
        port=MQTT_PORT,
        username=MQTT_USER,
        password=MQTT_PASS,
        client_id=MQTT_CLIENT_ID + "-pub",
        keepalive=MQTT_KEEPALIVE,
    )
    await client.connect()
    _mqtt_client = client
    log.info("[MQTT-PUB] Connected to %s:%s", MQTT_HOST, MQTT_PORT)
    return client


async def disconnect_client():
    global _mqtt_client
    if _mqtt_client is not None:
        try:
            await _mqtt_client.disconnect()
        except Exception:
            pass
        finally:
            _mqtt_client = None
        log.info("[MQTT-PUB] Disconnected")


async def publish_mqtt(
    topic: str,
    message: Dict[str, Any],
    *,
    qos: int = 1,
    retain: bool = False,
    max_retries: int = 5,
    base_delay: float = 0.5,
):
    global seq_number, _mqtt_client
    seq_number += 1

    body = dict(message)
    body["seq"] = seq_number
    payload = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")

    attempt = 0
    log.info("[MQTT-PUB] Try publishing to %s", topic)
    while True:
        try:
            async with mqtt_lock:
                client = await _connect_client()
                await client.publish(topic, payload, qos=qos, retain=retain)
            log.info(
                "[MQTT-PUB] Published to %s (qos=%s, retain=%s, seq=%s)",
                topic,
                qos,
                retain,
                seq_number,
            )
            return
        except MqttError as e:
            log.warning("[MQTT-PUB] Publish error: %s", e)
            async with mqtt_lock:
                if _mqtt_client:
                    try:
                        await _mqtt_client.disconnect()
                    except Exception:
                        pass
                    _mqtt_client = None
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(
                    f"[MQTT-PUB] Failed after {max_retries} retries: {e}"
                )
            delay = base_delay * (2 ** (attempt - 1))
            log.info("[MQTT-PUB] Retrying in %.1fs...", delay)
            await asyncio.sleep(delay)
