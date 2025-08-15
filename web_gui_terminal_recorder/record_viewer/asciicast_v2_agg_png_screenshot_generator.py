import agg_python_bindings
import tempfile
import os

input_file = "./vim_rectangle_selection_test/vim_rectangle_selection.cast"

with tempfile.TemporaryDirectory() as tmpdir:
    agg_python_bindings.load_asciicast_and_save_png_screenshots(input_file, png_write_dir = tmpdir)
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