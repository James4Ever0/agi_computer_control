import argparse
import threading
import json

# we need three processes/threads

# one for screenshot, using mss, taking one and save to jpeg every one second

# one for keyboard events, using pynput

# one for mouse events, using pynput

# the entire thing will be saved under a specific directory

# file handles will be closed within the "finally" code block

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    return parser

def screenshot_worker(output_dir:str):
    import mss
    import time
    import os
    import pathlib
    output_dir = os.path.join(output_dir, "screenshot")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    with mss.mss() as sct:
        while True:
            time.sleep(1)
            timestamp = time.time()
            screenshot = sct.shot(output=os.path.join(output_dir, "screenshot_%s.png" % timestamp))

def keyboard_worker(output_file:str):
    from pynput import keyboard
    import time
    with open(output_file, "a+") as f:
        def on_press(key):
            if type(key) != str:
                key = str(key)
            f.write(json.dumps(dict(event="key_press", key=key, timestamp=time.time())) + "\n")
            f.flush()


        def on_release(key):
            if type(key) != str:
                key = str(key)
            f.write(json.dumps(dict(event="key_release", key=key, timestamp=time.time())) + "\n")
            f.flush()

        with keyboard.Listener(on_press=on_press, on_release=on_release) as keyboard_listener:
            keyboard_listener.join()


def mouse_worker(output_file:str):
    from pynput import mouse
    import time
    with open(output_file, "a+") as f:
        def on_move(x: int, y: int):
            f.write(json.dumps(dict(event="mouse_move", x=x, y=y, timestamp=time.time())) + "\n")
            f.flush()


        def on_click(x: int, y: int, button: mouse.Button, pressed: bool):
            f.write(json.dumps(dict(event="mouse_click", x=x, y=y, button=str(button), pressed=pressed, timestamp=time.time())) + "\n")
            f.flush()


        def on_scroll(x: int, y: int, dx: int, dy: int):
            f.write(json.dumps(dict(event="mouse_scroll", x=x, y=y, dx=dx, dy=dy, timestamp=time.time())) + "\n")
            f.flush()


        with mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
            listener.join()


def keyboard_and_mouse_worker(output_dir:str):
    import os
    
    mouse_output_file = os.path.join(output_dir, "mouse.log")
    keyboard_output_file = os.path.join(output_dir, "keyboard.log")
    
    mouse_thread = threading.Thread(target=mouse_worker, args=(mouse_output_file,), daemon=True)
    keyboard_thread = threading.Thread(target=keyboard_worker, args=(keyboard_output_file,), daemon=True)

    mouse_thread.start()
    keyboard_thread.start()

    for thread in [mouse_thread, keyboard_thread]:
        thread.join()
    
def main():
    import pathlib
    parser = argument_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    # create the directory first
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    screenshot_thread = threading.Thread(target=screenshot_worker, args=(output_dir,), daemon=True)
    keyboard_and_mouse_thread = threading.Thread(target=keyboard_and_mouse_worker, args=(output_dir,), daemon=True)
    print("Starting threads...")
    try:
        screenshot_thread.start()
        keyboard_and_mouse_thread.start()
        for thread in [screenshot_thread, keyboard_and_mouse_thread]:
            thread.join()
    finally:
        print("Worker GUI: Stopping threads...")


if __name__ == "__main__":
    main()