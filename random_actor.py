# you generate states.jsonl.
# must include all possible states.
# try to match the distribution?

# i think it is kind of like monkey.js (aka monkey testing)?
# it's better to add some kind of randomness, or "experienced learner" responsible for generating new data, to overcome the shortage of imagination and possibilities.

# virtualbox unattended installation:
# vboxuser:changeme

# connect via openai universe (vnc)

# you can setup initial desktop environments, just like yours, using automated scripts.
# you perform your actions randomly, inject actions while the bot is acting alone.

import pynput

# what is the screen limit
import mss

screenshot_factory = mss.mss()

mon_0 = screenshot_factory.monitors[0]
# print(mon_0)

# 1280x800? width & height, dict
special_key_names = [
    "alt",
    "alt_l",
    "alt_r",
    "alt_gr",
    "backspace",
    "caps_lock",
    "cmd",
    "cmd_l",
    "cmd_r",
    "ctrl",
    "ctrl_l",
    "ctrl_r",
    "delete",
    "down",
    "end",
    "enter",
    "esc",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
    "f12",
    "f13",
    "f14",
    "f15",
    "f16",
    "f17",
    "f18",
    "f19",
    "f20",
    "home",
    "left",
    "page_down",
    "page_up",
    "right",
    "shift",
    "shift_l",
    "shift_r",
    "space",
    "tab",
    "up",
    "media_play_pause",
    "media_volume_mute",
    "media_volume_down",
    "media_volume_up",
    "media_previous",
    "media_next",
    "insert",
    "menu",
    "num_lock",
    "pause",
    "print_screen",
    "scroll_lock",
]

special_keys = []
missing_special_key_names = []
# on macOS? or MacBook?
# MISSING SPECIAL KEYS? ['insert', 'menu', 'num_lock', 'pause', 'print_screen', 'scroll_lock']

for name in special_key_names:
    try:
        key = pynput.keyboard.Key.__dict__[name]
        special_keys.append(key)
    except:
        missing_special_key_names.append(name)
import sys

print("MISSING SPECIAL KEYS?", missing_special_key_names, file=sys.stderr)
mouse_buttons = [
    pynput.mouse.Button.left,
    pynput.mouse.Button.right,
    pynput.mouse.Button.middle,
]

normal_keys = [
    ",",
    ".",
    "/",
    ";",
    "'",
    "[",
    "]",
    "\\",
    "=",
    "-",
    "0",
    "9",
    "8",
    "7",
    "6",
    "5",
    "4",
    "3",
    "2",
    "1",
    "`",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]

import random

# use repr() for conversion, str() for pynput.keyboard.Key


def get_keyboard_action(action_type: str):
    key = random.choice(normal_keys + special_keys)
    # pressed = random.choice([True, False])
    # action_type = "key_press" if pressed else "key_release"
    key_repr = repr(key) if type(key) == str else str(key)
    return (action_type, key_repr)


def get_random_location(mode=0):
    return (
        random.randint(mon_0["width"] * mode, mon_0["width"]),
        random.randint(mon_0["height"] * mode, mon_0["height"]),
    )


def get_mouse_move_action():
    return ("mouse_move", get_random_location())


def get_mouse_click_action():
    button = str(random.choice(mouse_buttons))
    pressed = random.choice([True, False])
    return ("mouse_click", (*get_random_location(), button, pressed))


def get_mouse_scroll_action():
    return ("mouse_scroll", (*get_random_location(), *get_random_location(mode=1)))


# mouse_move, mouse_click, mouse_scroll
action_types = ["mouse_move", "mouse_click", "mouse_scroll", "key_press", "key_release"]
keyboard_actions = ["key_press", "key_release"]
mouse_actions_mapping = {
    "mouse_move": get_mouse_move_action,
    "mouse_click": get_mouse_click_action,
    "mouse_scroll": get_mouse_scroll_action,
}
import json

if __name__ == "__main__": # you may also emit the delay data.
    for _ in range(100):
        actions = []
        act_or_not = random.randint(0, 5)
        if act_or_not > 3:
            action_type = random.choice(action_types)
            if action_type in keyboard_actions:
                actions.append(get_keyboard_action(action_type))
            else:
                actions.append(mouse_actions_mapping[action_type]())
        # print(actions)
        print(json.dumps({"HIDEvents": actions}))
