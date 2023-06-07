hid_timestamp_path = "recordings/2023-06-02T07_59_45.711256/hid_timestamps.json"

video_timestamp_path = "recordings/2023-06-02T07_59_45.711256/video_timestamps.json"

import json

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

hid_timestamp = load_json(hid_timestamp_path)
video_timestamp = load_json(video_timestamp_path)

import numpy as np

hidseq = np.zeros(shape=(2, len(hid_timestamp)))
hidseq[0] = np.array(range(len(hid_timestamp)))

videoseq = np.zeros(shape=(2, len(video_timestamp)))
videoseq[1] = np.array(range(len(video_timestamp)))

seq = np.hstack((hidseq, videoseq))
print("SEQ SHAPE?", seq.shape)

timeseq = np.array(hid_timestamp+video_timestamp)
sorted_indexes = np.argsort(timeseq)

sorted_seq = seq[:, sorted_indexes]
print(sorted_seq)