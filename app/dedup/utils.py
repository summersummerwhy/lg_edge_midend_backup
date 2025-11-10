def build_id_map(prv_alerts):
    id_map = {}
    for alert in prv_alerts:
        track_id = alert["payload"]["track_id"]
        alert_type = alert["payload"]["type"]
        prv_count = id_map.get(track_id, 0)

        if alert_type == "enter":
            id_map[track_id] = prv_count
        else:  # if alert_type == "exit":
            id_map[track_id] = prv_count + 1
    return id_map


def get_next_type_map(prv_alerts):
    nxt_type = {}
    for alert in prv_alerts:
        track_id = alert["payload"]["track_id"]
        alert_type = alert["payload"]["type"]

        if alert_type == "enter":
            nxt_type[track_id] = "exit"
        else:  # if alert_type == "exit":
            nxt_type[track_id] = "enter"
    return nxt_type
