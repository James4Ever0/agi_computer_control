# Web GUI/Terminal Recorder

<!-- TODO: use ducktrack for gui recording (https://github.com/TheDuckAI/DuckTrack) -->
<!-- TODO: follow osworld data curation format (https://os-world.github.io/explorer.html) -->

## Intro

This is a web GUI/terminal recorder. It can record the GUI/terminal input/output and save it to a file.

Has some buttons for controlling recording, a textarea for specifying output filename and descriptions, and an iframe for novnc/ttyd interactions.

The web server launches two docker containers (workers) running ttyd/novnc and recording the terminal/GUI input/output.

## Video demo

[cybergod_terminal_gui_recorder_demo.webm](https://github.com/user-attachments/assets/620e4625-38ca-4c83-ae35-410ef008c640)

## Usage

1. Prepare the images for the workers:

    Build the image locally:

    ```bash
    # building base images
    bash build_docker_image_worker_terminal.sh
    bash build_docker_image_worker_gui.sh

    # building remote connection base images
    bash build_docker_image_worker_terminal_remote_base.sh
    bash build_docker_image_worker_gui_remote_base.sh
    ```

    Or pull prebuilt images from Docker Hub:

    ```bash
    # pulling base images
    docker pull agile4im/cybergod_worker_gui
    docker pull agile4im/cybergod_worker_terminal

    docker tag agile4im/cybergod_worker_gui cybergod_worker_gui
    docker tag agile4im/cybergod_worker_terminal cybergod_worker_terminal

    # pulling images for remote connection
    docker pull agile4im/cybergod_worker_gui:remote-base
    docker pull agile4im/cybergod_worker_terminal:remote-base

    docker tag agile4im/cybergod_worker_gui:remote-base cybergod_worker_gui:remote-base
    docker tag agile4im/cybergod_worker_terminal:remote-base  cybergod_worker_terminal:remote-base 
    ```

2. Install the requirements for the web server:
    ```bash
    pip3 install -r server_webui_requirements.txt
    ```
3. Start the web server:
    ```bash
    bash serve.sh
    ```
4. Open the web GUI in a browser:
    ```
    http://localhost:9001
    ```
5. Start recording by clicking the "Start" button.
6. Stop recording by clicking the "Stop" button.
7. If you are using remote recorders, remember to set the correct connection info before starting the recording. (VNC for GUI, SSH for terminal)
8. Remember to enter the description for the recording in the textarea before stopping the recording.

## Notes

- The web server is built using FastAPI.
- The workers are built using Docker.
- The workers use ttyd/novnc to interact with the terminal/GUI.
- The workers use asciinema/mss/pynput to record the terminal/GUI input/output.
- The recorded sessions are saved to `./record` directory.
- The GUI remote recorder is using TigerVNC.
- The terminal remote recorder is using SSH.
- The terminal in GUI remote recorder is using lxterminal and SSH.
- Currently, only one recording can be done at a time.
- You need to have the following free ports on the host machine: 9001 (webui), 8080 (ttyd), 8081 (novnc), 8950 (vnc)

## Recording Format

Common file contents:

- `description.txt`: Description of the recording. Pure text.

- `begin_recording.txt`: Timestamp of when the recording started. JSON format.

    Example: `{"timestamp": <timestamp: float>, "event": "begin_recording"}`

- `stop_recording.txt`: Timestamp of when the recording stopped. JSON format. 

    Example: `{"timestamp": <timestamp: float>, "event": "stop_recording"}`

### GUI Recording

File structure:

```
./record/gui/gui_record_%Y%m%d_%H%M%S
    ├── description.txt
    ├── begin_recording.txt
    ├── stop_recording.txt
    ├── screenshot
    │   └── screenshot_<timestamp: float>.png
    ├── mouse.log
    └── keyboard.log
```

Unique file contents:

- `screenshot`: Directory containing screenshots (PNG files) of the GUI. Each screenshot is named with a timestamp.
- `mouse.log`: Log of mouse movements. Each line is a JSON object containing the timestamp and mouse event.
  
  Examples:
    - `{"event": "mouse_move", "x": <x: int>, "y": <y: int>, "timestamp": <timestamp: float>}`
    - `{"event": "mouse_click", "x": <x: int>, "y": <y: int>, "button": <button: str>, "pressed": <pressed: bool>, "timestamp": <timestamp: float>}`
    - `{"event": "mouse_scroll", "x": <x: int>, "y": <y: int>, "dx": <dx: int>, "dy": <dy: int>,  "timestamp": <timestamp: float>}`
- `keyboard.log`: Log of keyboard input. Each line is a JSON object containing the timestamp and keyboard event.

  Examples:
    - `{"event": "key_press", "key": <key: str>, "timestamp": <timestamp: float>}`
    - `{"event": "key_release", "key": <key: str>, "timestamp": <timestamp: float>}`

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

- `terminal.cast`:
    
    Log of terminal input and output ([asciinema v2 format](https://docs.asciinema.org/manual/asciicast/v2/)).
    
    The first line is a JSON object storing metadata about the recording.

    `{"version": 2, "width": <width: int>, "height": <height: int>, "timestamp": <timestamp: int>, "idle_time_limit": <idle_time_limit: float>, "env": <env: dict>, "title": <title: str>}`
    
    Each subsequent line is a JSON object containing the timestamp and terminal event. The terminal worker is configured to distinguish input from output.

    - input: `[<relative_timestamp: float>, "i", <content: str>]`
    - output: `[<relative_timestamp: float>, "o", <content: str>]`


## TODO

- [ ] Add support for multiple recordings at the same time.
- [ ] Add screenshot metadata saving to GUI recordings.
- [ ] Make terminal recordings fixed to 80x25 size.
- [ ] Implement GUI/terminal record viewer.
- [ ] Implement GUI/tetminal executor and replayer.
- [ ] Implement remote VNC recording by building a custom VNC client with input recording capabilities (for example, open the VNC client without window manager in x11vnc-desktop, and pass all input events into the client; or record noVNC client input events using client-side javascript or electron).
- [x] Implement terminal in GUI recorder.
- [x] Implement remote GUI recorder.
- [x] Implement remote terminal recorder.
- [x] Implement remote terminal in GUI recorder.