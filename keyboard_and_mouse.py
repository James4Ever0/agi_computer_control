from pynput import keyboard
import time

States = []

# import mss
# screenshot_factory = mss.mss()

from config import timestep  # this will be sufficient?

# problem is the windows zooming factor.
# is it really the problem?

# make it short since we cannot learn too many things.


def on_press(key):
    if type(key) != str:
        key = str(key)
    States.append({"HIDEvent": ("key_press", key), "timeStamp": time.time()})


def on_release(key):
    if type(key) != str:
        key = str(key)
    States.append({"HIDEvent": ("key_release", key), "timeStamp": time.time()})


keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
keyboard_listener.start()
from pynput import mouse


def on_move(x: int, y: int):
    States.append({"HIDEvent": ("mouse_move", [x, y]), "timeStamp": time.time()})


def on_click(x: int, y: int, button: mouse.Button, pressed: bool):
    States.append(
        {
            "HIDEvent": ("mouse_click", [x, y, str(button), pressed]),
            "timeStamp": time.time(),
        }
    )


def on_scroll(x: int, y: int, dx: int, dy: int):
    States.append(
        {"HIDEvent": ("mouse_scroll", [x, y, dx, dy]), "timeStamp": time.time()}
    )


# # ...or, in a non-blocking fashion:
listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
listener.start()

# you may start that non-blocking. start some looping-forever thread for writing states to file.
import time

# import pyautogui
# import datetime

loopCount = 500

import jsonlines

print("RECORDING START")
from config import filePath

# world_start = datetime.datetime.now()

# run mss in another thread? with lock? dead lock?

# generate two separate file?

with jsonlines.open(filePath, "w") as w:
    for loopIndex in range(loopCount):
        time.sleep(timestep)
        # as for screenshot, use mss instead of screenshot.
        #     screenshot = pyautogui.screenshot()
        # shall you mark the time here.
        # image = screenshot_factory.grab(screenshot_factory.monitors[0])
        # imagebytes = image.raw
        # imagePath = f"{imagedir}/{loopIndex}.raw" # no compression needed.
        # with open(imagePath, 'wb') as f:
        #     f.write(imagebytes)
        # timeDelay = (datetime.datetime.now() - world_start).total_seconds()
        # screenshot_factory.shot(output=imagePath)
        for state in States:
        # state = dict(
        #     # States=States,
        #     HIDEvent=HIDEvent,
        #     timeStamp=timeStamp
        #     # imagePath=imagePath,
        #     # timeDelay=timeDelay
        # )  # also the image!
            print("STATE?", state)
            w.write(state)
        States = []
        # mouseloc = []
