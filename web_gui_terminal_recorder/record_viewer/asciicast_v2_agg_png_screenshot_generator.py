# TODO: write image height and width to image filename
# TODO: do the same for gui recorder (mss)

import agg_python_bindings
import os
import json
from PIL import Image

# TODO: include the description file and add to the event stream header

input_file = "./vim_rectangle_selection_test/vim_rectangle_selection.cast"
output_path = "./cybergod_terminal_events.jsonl"
screenshot_path_relative = "./screenshots"
screenshot_path_absolute = os.path.join(
    os.path.dirname(os.path.abspath(output_path)), screenshot_path_relative
)

os.makedirs(screenshot_path_absolute, exist_ok=True)

screenshot_path_absolute = os.path.realpath(screenshot_path_absolute)
# read the cast file in our way first, convert to timeline-aligned events, then merge screenshot events with text events

# the cast file could be of three types:
# v1, v2, v3
# v1 is json. v2 and v3 are jsonl
# format reference:
# v1: https://docs.asciinema.org/manual/asciicast/v1/
# v2: https://docs.asciinema.org/manual/asciicast/v2/
# v3: https://docs.asciinema.org/manual/asciicast/v3/
# notice, parsing error could happen in pyo3 bindings (panic) or python code

text_events = []
event_abbr_mapping = dict(i="input", o="output", m="marker", r="resize", x="exit")

# resize is supported at v2
# exit is supported at v3

if input_file.endswith(".json"):
    with open(input_file, "r") as f:
        data = json.load(f)  # could be JSONDecodeError
        stdout_data = data.pop("stdout")
        asciicast_header = data
        for it in stdout_data:
            event_timestamp, event_payload = it
            text_events.append(
                dict(timestamp=event_timestamp, type="output", payload=event_payload)
            )
elif input_file.endswith(".cast"):
    with open(input_file, "r") as f:
        asciicast_header = None
        text_events = []
        for line in f.readlines():
            if asciicast_header is None:
                asciicast_header = json.loads(line)
            else:
                event_data = json.loads(line)
                event_timestamp, event_type_abbrivation, event_payload = event_data
                # convert to cybergod terminal event format
                # append to events
                event_type_fullname = event_abbr_mapping.get(
                    event_type_abbrivation, "unknown"
                )
                text_events.append(
                    dict(
                        timestamp=event_timestamp,
                        type=event_type_fullname,
                        type_abbrivation=event_type_abbrivation,
                        payload=event_payload,
                    )
                )
else:
    # file extension unknown
    raise ValueError("File '%s' does not end with '.json' or '.cast'" % input_file)

asciicast_version = asciicast_header["version"]
if asciicast_version not in [1]:
    print(
        "Warning: You are parsing asciinema v%s which may contain event 'resize'"
        % asciicast_version
    )

try:
    agg_python_bindings.load_asciicast_and_save_png_screenshots(
        input_file, png_write_dir=screenshot_path_absolute
    )
except BaseException as e:
    print("Possible panic from PyO3:", repr(e))
    print("Failed to extract screenshots from asciicast file")
print("Terminal screenshots extraction complete")
# now list all files in the tmpdir
png_filenames = os.listdir(screenshot_path_absolute)
# parse the file names
screenshot_events = []
for it in png_filenames:
    png_filepath = os.path.join(screenshot_path_absolute, it)
    img = Image.open(png_filepath)
    width, height = img.width, img.height
    timestamp_str = it.split("_")[-1][: -len(".png")]
    timestamp_float = float(timestamp_str)
    screenshot_events.append(
        dict(
            timestamp=timestamp_float,
            type="screenshot",
            screenshot_path=os.path.join(screenshot_path_relative, it),
            size=dict(height=height, width=width),
        )  # TODO: make the screenshot path relative to the generated event stream file (jsonl)
    )

screenshot_events.sort(key=lambda x: x["timestamp"])

screen_size = screenshot_events[0]["size"]
screen_size_changed = len([it for it in text_events if it['type'] == "resize"]) != 0

overall_events = screenshot_events + text_events
overall_events.sort(key=lambda x: x["timestamp"])

# get the first screenshot and obtain dimension aspects

# prepare and write output
output_header = dict(
    format="cybergod_terminal_record",
    version=1,
    asciicast_version=asciicast_version,
    header_data=asciicast_header,
    screen_size=screen_size,
    screen_size_changed=screen_size_changed
)

with open(output_path, "w+") as f:
    # write header first
    f.write(json.dumps(output_header) + "\n")
    for it in overall_events:
        f.write(json.dumps(it) + "\n")

print("Output write to:", output_path)
