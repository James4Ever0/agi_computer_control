# just to play the recorded events out.
# not doing anything beyond that.

# special keys are not recorded. what a shame.
import ast
import os

# what about "ctrl + /" ?


def unshift(key):
    lower_keycodes = {
        "A": "a",
        "B": "b",
        "C": "c",
        "D": "d",
        "E": "e",
        "F": "f",
        "G": "g",
        "H": "h",
        "I": "i",
        "J": "j",
        "K": "k",
        "L": "l",
        "M": "m",
        "N": "n",
        "O": "o",
        "P": "p",
        "Q": "q",
        "R": "r",
        "S": "s",
        "T": "t",
        "U": "u",
        "V": "v",
        "W": "w",
        "X": "x",
        "Y": "y",
        "Z": "z",
    }
    unshift_keycodes = {
        "!": "1",
        "@": "2",
        "#": "3",
        "$": "4",
        "%": "5",
        "^": "6",
        "&": "7",
        "*": "8",
        "(": "9",
        ")": "0",
        "_": "-",
        "+": "=",
        "{": "[",
        "}": "]",
        "|": "\\",
        ":": ";",
        '"': "'",
        "<": ",",
        ">": ".",
        "?": "/",
        "~": "`",
    }
    ctrl_keycodes = {
        "\x01": "a",
        "\x02": "b",
        "\x03": "c",
        "\x04": "d",
        "\x05": "e",
        "\x06": "f",
        "\x07": "g",
        "\x08": "h",
        "\t": "i",
        "\n": "j",
        "\x0b": "k",
        "\x0c": "l",
        "\r": "m",
        "\x0e": "n",
        "\x0f": "o",
        "\x10": "p",
        "\x11": "q",
        "\x12": "r",
        "\x13": "s",
        "\x14": "t",
        "\x15": "u",
        "\x16": "v",
        "\x17": "w",
        "\x18": "x",
        "\x19": "y",
        "\x1a": "z",
        "<219>": "[",
        "<221>": "]",
        "<189>": "-",
        "<187>": "=",
        "<192>": "`",
        "<48>": "0",
        "<49>": "1",
        "<50>": "2",
        "<51>": "3",
        "<52>": "4",
        "<53>": "5",
        "<54>": "6",
        "<55>": "7",
        "<56>": "8",
        "<57>": "9",
        "<220>": "\\",
        "<186>": ";",
        "<222>": "'",
        "<188>": ",",
        "<190>": ".",
        "<191>": "/",
    }
    key = unshift_keycodes.get(
        key, ctrl_keycodes.get(key, lower_keycodes.get(key, key))
    )
    return key


# how to play that?
# using jsonl?
# timestep = 0.01
from config import timestep

# zoom_factor = 1.75
# you can safely ignore the zoom factor once using this in the reference:
# https://pynput.readthedocs.io/en/latest/mouse.html#monitoring-the-mouse
import os

# main problem is the hotkey. but the modifiers are certain.
# you can know that for sure.

if os.name == "nt":
    import ctypes

    PROCESS_PER_MONITOR_DPI_AWARE = 2
    ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)

# better use pyautogui for mouse?
import jsonlines
import time
import pynput
import pyautogui

# filePath = "states.jsonl"
from config import filePath

with jsonlines.open(filePath) as r:
    stateList = list(r.iter())
# import math
keyboard_controller = pynput.keyboard.Controller()
mouse_controller = pynput.mouse.Controller()

mouse_buttons = [
    pynput.mouse.Button.left, pynput.mouse.Button.right, pynput.mouse.Button.middle
]
mouse_button_states = {button: False for button in mouse_buttons}

for state in stateList:
    time.sleep(timestep)
    # perform actions.
    HIDEvents = state["HIDEvents"]
    # you need to preserve the order. dump all events into one single list.
    for action_type, action_args in HIDEvents:
        print("ACTION?", action_type, action_args)
        if action_type == "key_press":
            if not action_args.startswith("Key."):
                keycode = unshift(
                    action_args if action_args.startswith("<")
                    and action_args.endswith(">") else ast.literal_eval(action_args)
                )
                pyautogui.write(keycode)

            else:
                keyboard_controller.press(
                    pynput.keyboard.Key.__dict__[action_args.split(".")[-1]]
                )
        elif action_type == "key_release":
            if action_args.startswith("Key."):
                keyboard_controller.release(
                    pynput.keyboard.Key.__dict__[action_args.split(".")[-1]]
                )
        elif action_type == "mouse_move":
            x, y = action_args
            # x = math.floor(x/zoom_factor)
            # y = math.floor(y/zoom_factor)
            mouse_controller.position = (x, y)
        elif action_type == "mouse_click":
            x, y, button, pressed = action_args
            # x = math.floor(x/zoom_factor)
            # y = math.floor(y/zoom_factor)
            button = pynput.mouse.Button.__dict__[button.split(".")[-1]]
            mouse_button_states[button] = pressed
            mouse_controller.position = (x, y)
            if pressed:
                mouse_controller.press(button)
            else:
                mouse_controller.click(button)
        elif action_type == "mouse_scroll":
            x, y, dx, dy = action_args

            # x = math.floor(x/zoom_factor)
            # y = math.floor(y/zoom_factor)

            # dx = math.floor(dx/zoom_factor)
            # dy = math.floor(dy/zoom_factor)

            mouse_controller.position = (x, y)
            mouse_controller.scroll(dx, dy)
        else:
            raise Exception("Unknown action type: {}".format(action_type))

# after all the havok, you should not leave the mouse button pressed, and you should not leave any button onhold.

modifier_keys = ["alt", "alt_gr", "ctrl", "shift"]
for modifier_key in modifier_keys:
    if keyboard_controller.__getattribute__(f"{modifier_key}_pressed"):
        keyboard_controller.release(pynput.keyboard.Key.__dict__[modifier_key])

for button in mouse_buttons:
    if mouse_button_states[button]:  # pressed, not released yet.
        mouse_controller.release(button)
