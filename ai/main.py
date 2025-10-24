from inference import inference
from tracker import tracker
import utils
import mqtt

import cv2


def visualize(image, box, color):
    x1, y1, x2, y2 = box

    out = image.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 4)
    return out


def track_image(image, image_format):
    objects = inference(image)
    result = tracker(objects)

    track_list = result["objects"]
    old_track_ids = result["old_ids"]
    new_track_ids = result["new_ids"]
    # 셋 다 {"box": xyxy, "track_id": track_id} 형태
    # track_list는 dict, old/new_track_ids는 list

    payloads = []

    for obj in old_track_ids:
        box = obj["box"]
        old_id = obj["track_id"]

        old_image = visualize(image, box, (0, 0, 255))
        payloads.append(
            {
                "type": "exit",
                "track_id": old_id,
                "image": {
                    "format": image_format,
                    "width": old_image.shape[1],
                    "height": old_image.shape[0],
                    "data_b64": utils.to_base64(old_image, image_format),
                },
            }
        )

    for obj in new_track_ids:
        box = obj["box"]
        new_id = obj["track_id"]

        new_image = visualize(image, box, (0, 255, 0))
        payloads.append(
            {
                "type": "enter",
                "track_id": new_id,
                "image": {
                    "format": image_format,
                    "width": new_image.shape[1],
                    "height": new_image.shape[0],
                    "data_b64": utils.to_base64(new_image, image_format),
                },
            }
        )

    return payloads


if __name__ == "__main__":
    DEVICE = "topst"

    while True:
        recv_message = mqtt.wait_for_mqtt()

        timestamp = recv_message["ts"]
        image = recv_message["payload"]["data_b64"]
        image_format = recv_message["payload"]["format"]

        image = utils.from_base64(image, image_format)
        payloads = track_image(image, image_format)

        for payload in payloads:
            message = {"device": DEVICE, "ts": timestamp, "payload": payload}
            mqtt.publish_mqtt(f"topst/{DEVICE}/ai", message)
