the script ./gui_record_folder_to_jsonl_event_stream.py will rearrange gui recording into a unified event stream jsonl file.

input example: ../record/gui/gui_record_20250805_233855
output example: ./gui_jsonl_event_stream.jsonl

format specification:

first line is the metadata, in json format

metadata format:

{
    "file_format": "cybergod_gui_record",
    "version": "1",
    "screen_size": {
        "height": <height: int>,
        "width": <width: int>
    },
    "begin_recording": <begin_recording_timestamp: float>,
    "stop_recording": <stop_recording_timestamp: float>,
    "duration": <stop_recording_timestamp: float> - <begin_recording_timestamp: float>,
    "description": <description: str>,
    "basedir": <basedir: str>
}

subsequent lines are events, in json format

event format:

{
    "timestamp": <timestamp : float>,
    "event_source": "screenshot" or "keyboard" or "mouse",
    "event_type": "screenshot" or "key_press" or "key_release" or "mouse_move" or "mouse_click" or "mouse_scroll",
    "event_data": **event_source dependent data**
}

event data format:

    for screenshot events:

    {
        "screenshot_file": <screenshot_filename: str>
    }

    for keyboard events:

    {
        "event": "key_press" or "key_release",
        "key": <key: str>,
    }

    for mouse events:

    {"event": "mouse_move", "x": <x: int>, "y": <y: int>}

    {"event": "mouse_click", "x": <x: int>, "y": <y: int>, "button": <button: str>, "pressed": <pressed: bool>}

    {"event": "mouse_scroll", "x": <x: int>, "y": <y: int>, "dx": <dx: int>, "dy": <dy: int>}