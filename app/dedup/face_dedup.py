from collections import deque
import logging

log = logging.getLogger(__name__)

# Global state
track_face_map = {}  # track_id -> face_id
pending_exits = (
    deque()
)  # [{"ts": int, "alert": dict, "face_id": str, "expire_ts": int}]
EXIT_DELAY_MS = 1000


def face_dedup(alerts):
    global track_face_map, pending_exits

    output_alerts = []

    for alert in alerts:
        current_ts = alert["ts"]

        # 1. 만료된 퇴장 알림 방출
        while pending_exits and pending_exits[0]["expire_ts"] <= current_ts:
            expired = pending_exits.popleft()
            output_alerts.append(expired["alert"])

        payload = alert["payload"]
        alert_type = payload["type"]
        track_id = payload["track_id"]

        if alert_type == "enter":
            face_id = payload.get("face_id")

            if face_id and face_id != "unknown":
                track_face_map[track_id] = face_id

                # 1초 이내에 퇴장한 기록이 있는지 확인
                found_idx = -1
                for i, p_exit in enumerate(pending_exits):
                    if p_exit["face_id"] == face_id:
                        found_idx = i
                        break

                if found_idx != -1:
                    # 매칭되는 퇴장 알림 발견 -> 둘 다 무시
                    removed_exit = pending_exits[found_idx]
                    del pending_exits[found_idx]

                    log.info(
                        f"[FACE_DEDUP] Suppressed re-entry: face_id={face_id}, "
                        f"exit_track={removed_exit['alert']['payload']['track_id']}, "
                        f"enter_track={track_id}"
                    )
                    continue  # 현재 입장 알림 무시

            output_alerts.append(alert)

        elif alert_type == "exit":
            face_id = track_face_map.get(track_id)

            if face_id:
                # 퇴장 알림 버퍼링
                expire_ts = current_ts + EXIT_DELAY_MS
                pending_exits.append(
                    {
                        "alert": alert,
                        "face_id": face_id,
                        "expire_ts": expire_ts,
                    }
                )
                # 맵에서 제거 (버퍼에 face_id 저장됨)
                del track_face_map[track_id]
            else:
                # face_id 모르면 그냥 통과
                output_alerts.append(alert)
                if track_id in track_face_map:
                    del track_face_map[track_id]

        else:
            output_alerts.append(alert)

    return output_alerts
