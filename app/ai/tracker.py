track_list = {}


def tracker(objects):
    old_track_ids = track_list.copy()
    new_track_ids = []

    for obj in objects:
        track_id = obj["track_id"]

        if track_id in old_track_ids:
            del old_track_ids[track_id]
        else:
            new_track_ids.append(obj)

        track_list[track_id] = obj

    for old_id in old_track_ids.keys():
        del track_list[old_id]

    old_track_ids = list(old_track_ids.values())

    return {
        "objects": track_list,
        "old_ids": old_track_ids,
        "new_ids": new_track_ids,
    }
