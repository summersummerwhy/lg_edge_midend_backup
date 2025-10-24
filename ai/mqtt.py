seq_number = 0


def wait_for_mqtt():
    pass


def publish_mqtt(topic, message):
    global seq_number
    seq_number += 1
    message["seq"] = seq_number

    print(f"Publishing to topic {topic}: {message}")
