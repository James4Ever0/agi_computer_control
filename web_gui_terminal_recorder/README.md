## Intro

This is a web GUI/terminal recorder. It can record the GUI/terminal input/output and save it to a file.

Has some buttons for controlling recording, a textarea for specifying output filename and descriptions, and an iframe for novnc/ttyd interactions.

The Web server launches two Docker containers (workers) running ttyd/novnc and recording the terminal/GUI input/output.

## Usage

1. Build the images for the workers:
    ```
    bash build_docker_image_worker_terminal.sh
    bash build_docker_image_worker_gui.sh
    ```
2. Install the requirements for the web server:
    ```
    pip3 install -r server_webui_requirements.txt
    ```
3. Start the web server:
    ```
    bash serve.sh
    ```
4. Open the web GUI in a browser:
    ```
    http://localhost:9001
    ```
5. Start recording by clicking the "Start" button.
6. Stop recording by clicking the "Stop" button.
7. Remember to enter the description for the recording in the textarea before stopping the recording.

## Notes

- The web server is built using FastAPI.
- The workers are built using Docker.
- The workers use ttyd/novnc to interact with the terminal/GUI.
- The workers use asciinema/mss/pynput to record the terminal/GUI input/output.
- The recorded sessions are saved to `./record` directory.
- Currently, only one recording can be done at a time.

## Recording Format

Common file contents:

- `description.txt`: Description of the recording. Pure text.
- `begin_recording.txt`: Timestamp of when the recording started. JSON format. Example: `{"timestamp": <timestamp>, "event": "begin_recording"}`
- `stop_recording.txt`: Timestamp of when the recording stopped. JSON format. Example: `{"timestamp": <timestamp>, "event": "stop_recording"}`

### GUI Recording

File structure:

```
./record/gui/gui_record_%Y%m%d_%H%M%S
    ├── description.txt
    ├── begin_recording.txt
    ├── stop_recording.txt
    ├── screenshot
    │   ├── screenshot_<timestamp>.png
    ├── mouse.log
    └── keyboard.log
```

Unique file contents:

- `screenshot`: Directory containing screenshots of the GUI. Each screenshot is named with a timestamp.
- `mouse.log`: Log of mouse movements. Each line is a JSON object containing the timestamp and mouse event.
- `keyboard.log`: Log of keyboard input. Each line is a JSON object containing the timestamp and keyboard event.

### Terminal Recording

File structure:

```
./record/terminal/terminal_record_%Y%m%d_%H%M%S
    ├── description.txt
    ├── begin_recording.txt
    ├── stop_recording.txt
    └── terminal.cast
```

Unique file contents:

- `terminal.cast`: Log of terminal input and output (asciinema v2 format). The first line is a JSON object storing metadata about the recording. Each subsequent line is a JSON object containing the timestamp and terminal event.

## TODO

- [ ] Add support for multiple recordings at the same time.
- [ ] Add screenshot metadata saving to GUI recordings.