basePath = "recordings/2023-06-02T07_59_45.711256/"

hid_timestamp_path = f"{basePath}hid_timestamps.json"

video_timestamp_path = f"{basePath}video_timestamps.json"

import json


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

hid_timestamp = load_json(hid_timestamp_path)
video_timestamp = load_json(video_timestamp_path)

import numpy as np

hidseq = np.zeros(shape=(2, len(hid_timestamp)))-1
hidseq[0] = np.array(range(len(hid_timestamp)))

videoseq = np.zeros(shape=(2, len(video_timestamp)))-1
videoseq[1] = np.array(range(len(video_timestamp)))

seq = np.hstack((hidseq, videoseq))
print("SEQ SHAPE?", seq.shape)

timeseq = np.array(hid_timestamp+video_timestamp)
sorted_indexes = np.argsort(timeseq)

sorted_seq = seq[:, sorted_indexes].T
print(sorted_seq)

# now, attempt to parse them.

import cv2
import jsonlines

video_path = f"{basePath}video_record.mp4"
hid_rec_path = f"{basePath}hid_record.jsonl"

video_cap = cv2.VideoCapture(video_path)


success, frame = video_cap.read()
print(frame.shape) # (768, 1280, 3)