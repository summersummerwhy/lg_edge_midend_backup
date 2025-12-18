from collections import deque
import logging

log = logging.getLogger(__name__)

# Global state
pending_exits = (
    deque()
)  # [{"ts": int, "alert": dict, "face_id": str, "expire_ts": int}]
EXIT_DELAY_MS = 1000


def face_dedup(alerts):
    global pending_exits

    output_alerts = []

    for alert in alerts:
        current_ts = alert["ts"]

        # 1. 만료된 퇴장 알림 방출
        while pending_exits and pending_exits[0]["expire_ts"] <= current_ts:
            expired = pending_exits.popleft()
            output_alerts.append(expired["alert"])

        payload = alert["payload"]
        if payload.get("priority") == "low":
            output_alerts.append(alert)
            continue

        alert_type = payload["type"]
        track_id = payload["track_id"]

        if alert_type == "enter":
            face_id = payload.get("face_id")

            if face_id and face_id != "unknown":
                # 1초 이내에 퇴장한 기록이 있는지 확인
                found_idx = -1
                for i, p_exit in enumerate(pending_exits):
                    if p_exit["face_id"] == face_id and p_exit["alert"]["payload"]["priority"] != "low":
                        found_idx = i
                        break

                if found_idx != -1:
                    # 매칭되는 퇴장 알림 발견 -> 둘 다 무시
                    removed_exit = pending_exits[found_idx]
                    removed_exit["alert"]["payload"]["priority"] = "low"

                    alert["payload"]["priority"] = "low"
                    output_alerts.append(alert)

                    log.info(
                        f"[FACE_DEDUP] Suppressed re-entry: face_id={face_id}, "
                        f"exit_track={removed_exit['alert']['payload']['track_id']}, "
                        f"enter_track={track_id}"
                    )
                    continue  # 현재 입장 알림 무시

            output_alerts.append(alert)

        elif alert_type == "exit":
            face_id = payload.get("face_id")

            if face_id and face_id != "unknown":
                # 퇴장 알림 버퍼링
                expire_ts = current_ts + EXIT_DELAY_MS
                pending_exits.append(
                    {
                        "alert": alert,
                        "face_id": face_id,
                        "expire_ts": expire_ts,
                    }
                )
            else:
                # face_id 모르면 그냥 통과
                output_alerts.append(alert)

        else:
            output_alerts.append(alert)

    return output_alerts
