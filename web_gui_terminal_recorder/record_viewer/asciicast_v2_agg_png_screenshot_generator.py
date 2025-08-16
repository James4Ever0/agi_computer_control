import agg_python_bindings
import tempfile
import os
import json

input_file = "./vim_rectangle_selection_test/vim_rectangle_selection.cast"
output_path = "./cybergod_terminal_events.jsonl"


# read the cast file in our way first, convert to timeline-aligned events, then merge screenshot events with text events

# the cast file could be of three types:
# v1, v2, v3
# v1 is json. v2 and v3 are jsonl
# format reference: 
# v1: https://docs.asciinema.org/manual/asciicast/v1/
# v2: https://docs.asciinema.org/manual/asciicast/v2/
# v3: https://docs.asciinema.org/manual/asciicast/v3/
# notice, parsing error could happen in pyo3 bindings (panic) or python code

if input_file.endswith(".json"):
    with open(input_file, "r") as f:
        data = json.load(f) # could be JSONDecodeError
        asciicast_header = ...
        text_events = []
elif input_file.endswith(".cast"):
    with open(input_file, 'r') as f:
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
                event_type_fullname = ...
                text_events.append(dict(timestamp = event_timestamp))
else:
    # file extension unknown
    raise ValueError("File '%s' does not end with '.json' or '.cast'" % input_file)


with tempfile.TemporaryDirectory() as tmpdir:
    try:
        agg_python_bindings.load_asciicast_and_save_png_screenshots(input_file, png_write_dir = tmpdir)
    except BaseException as e:
        print("Possible panic from PyO3:", repr(e))
        print("Failed to extract screenshots from asciicast file")
    print("Terminal screenshots extraction complete")
    # now list all files in the tmpdir
    png_filenames = os.listdir(tmpdir)
    # parse the file names
    screenshot_events = []
    for it in png_filenames:
        timestamp_str = it.split("_")[-1][:-len(".png")]
        timestamp_float = float(timestamp_str)
        screenshot_events.append(dict(timestamp = timestamp_float, screenshot_path = it))
    screenshot_events.sort(key=lambda x: x['timestamp'])
    for it in screenshot_events:
        print(it)

overall_events = screenshot_events + text_events
overall_events.sort(key=lambda x: x['timestamp'])

# prepare and write output
output_header = dict()

with open(output_path, 'w+') as f:
    # write header first
    f.write(json.dumps(output_header)+"\n")
    for it in overall_events:
        f.write(json.dumps(it)+"\n")

print("Output write to:", output_path)