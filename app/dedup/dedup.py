from collections import deque

prv_alerts = deque()

MINUTE_IN_MS = 60 * 1000
MAX_ENTER_COUNT = 3


def update_id_map(id_map, alert):
    track_id = alert["payload"]["track_id"]
    alert_type = alert["payload"]["type"]
    prv_count = id_map.get(track_id, 0)

    if alert_type == "enter":
        id_map[track_id] = prv_count
    else:  # if alert_type == "exit":
        id_map[track_id] = prv_count + 1


def update_next_type(nxt_type, alert):
    track_id = alert["payload"]["track_id"]
    alert_type = alert["payload"]["type"]

    if alert_type == "enter":
        nxt_type[track_id] = "exit"
    else:  # if alert_type == "exit":
        nxt_type[track_id] = "enter"


def dedup(alerts):
    # 1분 이내 알림만 남김
    max_ts = alerts[-1]["ts"]
    min_ts = max_ts - MINUTE_IN_MS

    while prv_alerts and prv_alerts[0]["ts"] <= min_ts:
        prv_alerts.popleft()

    # 변수 초기화
    id_map = {}
    nxt_type = {}
    deduped_alerts = []

    for alert in prv_alerts:
        update_id_map(id_map, alert)
        # update_next_type(nxt_type, alert)

    for alert in alerts:
        # id 3번 이상 입/퇴장시 무시
        track_id = alert["payload"]["track_id"]
        id_count = id_map.get(track_id, 0)

        if id_count >= MAX_ENTER_COUNT:
            # continue # 원래 무시해야 하지만 테스트를 위해 priority 0 처리
            alert["payload"]["priority"] = 0

        """
        # 중복 입/퇴장 알림 무시
        alert_type = alert["payload"]["type"]
        expected_type = nxt_type.get(track_id, "enter")

        if alert_type != expected_type:
            # continue # 원래 무시해야 하지만 테스트를 위해 priority 0 처리
            alert["payload"]["priority"] = 0
        """

        if alert["payload"]["priority"] != 0:  # continue로 무시하면 필요없는 line
            update_id_map(id_map, alert)
            update_next_type(nxt_type, alert)

        deduped_alerts.append(alert)

    prv_alerts.extend(deduped_alerts)  # 알림리스트 업데이트
    return deduped_alerts
