def suppression(alerts):
    for alert in alerts:
        payload = alert.get("payload", {})
        face_id = payload.get("face_id")

        # 기본값: unknown/high
        pr = "high"

        # face_id가 있고 unknown이 아니면 known
        if face_id is not None and str(face_id).lower() != "unknown":
            pr = "medium"

        # (선택) exit는 알림 중요도 낮추고 싶으면 여기서 조절 가능
        if payload.get("type") == "exit":
            pr = "low"

        payload["priority"] = pr
        alert["payload"] = payload

    return alerts
