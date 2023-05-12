# import datetime
from utils import filepaths, check_redis_on, check_redis_off, TimestampedContext, set_redis_off_on_exception

import jsonlines
import time
from pynput import mouse
from utils import timestep  # this will be sufficient?
from pynput import keyboard

HIDEvents = []

set_redis_off_on_exception()

# problem is the windows zooming factor.
# is it really the problem?


def on_press(key):
    if type(key) != str:
        key = str(key)
    HIDEvents.append(("key_press", key))


def on_release(key):
    if type(key) != str:
        key = str(key)
    HIDEvents.append(("key_release", key))


keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
keyboard_listener.start()


def on_move(x: int, y: int):
    HIDEvents.append(("mouse_move", [x, y]))


def on_click(x: int, y: int, button: mouse.Button, pressed: bool):
    HIDEvents.append(("mouse_click", [x, y, str(button), pressed]))


def on_scroll(x: int, y: int, dx: int, dy: int):
    HIDEvents.append(("mouse_scroll", [x, y, dx, dy]))


# # ...or, in a non-blocking fashion:
listener = mouse.Listener(
    on_move=on_move, on_click=on_click, on_scroll=on_scroll)
listener.start()

# you may start that non-blocking. start some looping-forever thread for writing states to file.

# import pyautogui
# import datetime

# loopCount = 500


print("RECORDING START")

# world_start = datetime.datetime.now()

if check_redis_on():
    with TimestampedContext(filepaths.hid_timestamps) as t:
        with jsonlines.open(filepaths.hid_record, "w") as w:
            # for _ in range(loopCount):
            while check_redis_off() is False:
                time.sleep(timestep)
                # as for screenshot, use mss instead of screenshot.
                #     screenshot = pyautogui.screenshot()
                # shall you mark the time here.
                state = dict(HIDEvents=HIDEvents)  # also the image!
                print("STATE?", state)
                w.write(state)
                t.commit()
                HIDEvents = []
                mouseloc = []
    print("EXITING HID RECORDER.")
    print("SAVING HID RECORDS TO:", filepaths.hid_record)
else:
    raise Exception("HIDRecorder: Can't start. Redis signal is off.".upper())
