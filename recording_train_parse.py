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

sorted_seq = seq[:, sorted_indexes].T.astype(int)
# print(sorted_seq)

# now, attempt to parse them.

import cv2
import jsonlines

video_path = f"{basePath}video_record.mp4"
hid_rec_path = f"{basePath}hid_record.jsonl"

video_cap = cv2.VideoCapture(video_path)
# breakpoint()
# 318 frames? only got 266 timestamps!
frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("FRAME COUNT?", frame_count)

hid_data_list = []
with open(hid_rec_path, 'r') as f:
    jsonl_reader =  jsonlines.Reader(f)
    while True:
        try:
            hid_data = jsonl_reader.read()
            hid_data_list.append(hid_data)
        except:
            break

# maybe you should "yield" data through these iterators.

NO_CONTENT = -1
for hid_index, frame_index in sorted_seq:
    print(hid_index, frame_index)
    assert not all([e == NO_CONTENT for e in [hid_index, frame_index]]), "at least one type of content is active"
    assert not all([e != NO_CONTENT for e in [hid_index, frame_index]]), "cannot have two types of active content sharing the same index"
    if hid_index != NO_CONTENT:
        hid_data = hid_data_list[hid_index]
        print(hid_data)
    elif frame_index != NO_CONTENT:
        suc, frame = video_cap.read()
        assert suc, f"Video '{video_path}' failed to read frame #{frame_index} (index starting from zero)"
        print(frame.shape)
        cv2.imshow('win',frame)
        cv2.waitKey(1)
    else:
        raise Exception("Something impossible has happened.")

# breakpoint()
video_cap.release()
# success, frame = video_cap.read()
# print(frame.shape) # (768, 1280, 3)