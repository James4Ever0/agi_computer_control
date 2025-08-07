import os
import json

basedir = "../record/gui/gui_record_20250805_233855"

screenshot_dir = "%s/screenshot" % basedir
keyboard_log_file = "%s/keyboard.log" % basedir
mouse_log_file = "%s/mouse.log" % basedir

begin_recording_file = "%s/begin_recording.txt" % basedir
stop_recording_file = "%s/stop_recording.txt" % basedir

with open(begin_recording_file, "r") as f:
    begin_recording_data = json.loads(f.read())
    begin_recording_timestamp = begin_recording_data["timestamp"]

with open(stop_recording_file, "r") as f:
    stop_recording_data = json.loads(f.read())
    stop_recording_timestamp = stop_recording_data["timestamp"]

screenshot_filenames = os.listdir(screenshot_dir)

screenshot_log_data = []
for filename in screenshot_filenames:
    data = dict(
        screenshot_file="screenshot/%s" % filename,
        timestamp=float(filename[:-4].split("_")[-1]),
        event_source="screenshot",
    )
    screenshot_log_data.append(data)


keyboard_log_data = []
with open(keyboard_log_file, "r") as f:
    for line in f:
        data = json.loads(line)
        data["event_source"] = "keyboard"
        keyboard_log_data.append(data)

mouse_log_data = []
with open(mouse_log_file, "r") as f:
    for line in f:
        data = json.loads(line)
        data["event_source"] = "mouse"
        mouse_log_data.append(data)

# now merge these three things
merged_event_stream = screenshot_log_data + keyboard_log_data + mouse_log_data
merged_event_stream.sort(key=lambda x: x["timestamp"])

# now filter out events that happened after the stop recording timestamp
merged_event_stream = [event for event in merged_event_stream if event["timestamp"] <= stop_recording_timestamp and event["timestamp"] >= begin_recording_timestamp]

print("Begin recording timestamp: %s" % begin_recording_timestamp)
print("Stop recording timestamp: %s" % stop_recording_timestamp)

for event in merged_event_stream:
    print(event)

# TODO: load the metadata of the screenshot recording, such as the width and height of the screen


# optionally, save the merged event stream to a file

with open("merged_gui_event_stream.json", "w") as f:
    json.dump(merged_event_stream, f)

# TODO: save all events and metadata to a jsonl file, similar to asciicast v2 format
# design a metadata header like:
# {"file_format": "cybergod_gui_record", "version": "1", "screen_size": {"height": <height: int>, width: <width: int>}, "begin_recording": <begin_recording: float>, "stop_recording": <stop_recording: float>, "description": <description: str>, "basedir": "./"}