hid_timestamp_path = "recordings/2023-06-02T07_59_45.711256/hid_timestamps.json"

video_timestamp_path = "recordings/2023-06-02T07_59_45.711256/video_timestamps.json"

import json

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

hid_timestamp = load_json(hid_timestamp_path)
video_timestamp = load_json(video_timestamp_path)