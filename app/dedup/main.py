from .dedup import dedup
from .suppression import suppression

import logging

log = logging.getLogger(__name__)


def process_alerts(alerts):
    # 보낼 알림이 없는 경우 처리 X
    if not alerts:
        return

    alerts = sorted(alerts, key=lambda x: x["ts"])

    log.info(f"[DEDUP] Processing {len(alerts)} alerts")

    # process
    alerts = suppression(alerts)

    log.info(f"[DEDUP] Alerts after suppression: {len(alerts)} alerts")

    alerts = dedup(alerts)

    log.info(f"[DEDUP] Alerts after deduplication: {len(alerts)} alerts")

    return alerts


if __name__ == "__main__":
    sample_alerts = [
        {
            "device": "esp32-wroom-01",
            "ts": 10,
            "seq": 1,
            "payload": {
                "type": "enter",
                "track_id": 1,
                "image": {
                    "format": "jpeg",
                    "width": 640,
                    "height": 480,
                    "data_b64": "b64",
                },
            },
        },
        {
            "device": "esp32-wroom-01",
            "ts": 20,
            "seq": 2,
            "payload": {
                "type": "enter",
                "track_id": 2,
                "image": {
                    "format": "jpeg",
                    "width": 640,
                    "height": 480,
                    "data_b64": "b64",
                },
            },
        },
        {
            "device": "esp32-wroom-01",
            "ts": 30,
            "seq": 3,
            "payload": {
                "type": "exit",
                "track_id": 1,
                "image": {
                    "format": "jpeg",
                    "width": 640,
                    "height": 480,
                    "data_b64": "b64",
                },
            },
        },
        {
            "device": "esp32-wroom-01",
            "ts": 40,
            "seq": 4,
            "payload": {
                "type": "enter",
                "track_id": 1,
                "image": {
                    "format": "jpeg",
                    "width": 640,
                    "height": 480,
                    "data_b64": "b64",
                },
            },
        },
        {
            "device": "esp32-wroom-01",
            "ts": 50,
            "seq": 5,
            "payload": {
                "type": "exit",
                "track_id": 1,
                "image": {
                    "format": "jpeg",
                    "width": 640,
                    "height": 480,
                    "data_b64": "b64",
                },
            },
        },
        {
            "device": "esp32-wroom-01",
            "ts": 60,
            "seq": 6,
            "payload": {
                "type": "enter",
                "track_id": 1,
                "image": {
                    "format": "jpeg",
                    "width": 640,
                    "height": 480,
                    "data_b64": "b64",
                },
            },
        },
        {
            "device": "esp32-wroom-01",
            "ts": 70,
            "seq": 7,
            "payload": {
                "type": "exit",
                "track_id": 1,
                "image": {
                    "format": "jpeg",
                    "width": 640,
                    "height": 480,
                    "data_b64": "b64",
                },
            },
        },
        {
            "device": "esp32-wroom-01",
            "ts": 80,
            "seq": 8,
            "payload": {
                "type": "enter",
                "track_id": 1,
                "image": {
                    "format": "jpeg",
                    "width": 640,
                    "height": 480,
                    "data_b64": "b64",
                },
            },
        },
        {
            "device": "esp32-wroom-01",
            "ts": 90,
            "seq": 9,
            "payload": {
                "type": "exit",
                "track_id": 1,
                "image": {
                    "format": "jpeg",
                    "width": 640,
                    "height": 480,
                    "data_b64": "b64",
                },
            },
        },
        {
            "device": "esp32-wroom-01",
            "ts": 60050,
            "seq": 10,
            "payload": {
                "type": "enter",
                "track_id": 1,
                "image": {
                    "format": "jpeg",
                    "width": 640,
                    "height": 480,
                    "data_b64": "b64",
                },
            },
        },
        {
            "device": "esp32-wroom-01",
            "ts": 60051,
            "seq": 11,
            "payload": {
                "type": "exit",
                "track_id": 1,
                "image": {
                    "format": "jpeg",
                    "width": 640,
                    "height": 480,
                    "data_b64": "b64",
                },
            },
        },
        {
            "device": "esp32-wroom-01",
            "ts": 60061,
            "seq": 12,
            "payload": {
                "type": "enter",
                "track_id": 1,
                "image": {
                    "format": "jpeg",
                    "width": 640,
                    "height": 480,
                    "data_b64": "b64",
                },
            },
        },
        {
            "device": "esp32-wroom-01",
            "ts": 60090,
            "seq": 13,
            "payload": {
                "type": "exit",
                "track_id": 1,
                "image": {
                    "format": "jpeg",
                    "width": 640,
                    "height": 480,
                    "data_b64": "b64",
                },
            },
        },
    ]

    for alert in sample_alerts:
        result = process_alerts([alert])[0]
        result = {
            "ts": result["ts"],
            "type": result["payload"]["type"],
            "track_id": result["payload"]["track_id"],
            "priority": result["payload"]["priority"],
        }

        print(result)
