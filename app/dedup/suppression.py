def suppression(alerts):
    # TODO: 아는 사람인 경우 priority 낮춤

    for alert in alerts:
        alert["payload"]["priority"] = 2
        # TODO: 불필요한 알림인 경우가 사실상 없지 않나?
        #       -> UI랑 연계해서 사용자가 설정한 알림만 priority 울리게 하기?
        # alert_type = alert["payload"]["type"]
        # if (alert_type == "exit" and options.suppress_exit_alerts):
        #     alert["payload"]["priority"] = 0

    return alerts
