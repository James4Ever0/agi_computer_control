from pynput import keyboard


def on_press(key):
    if type(key) != str:
        key = str(key)
    print(("key_press", key))


def on_release(key):
    if type(key) != str:
        key = str(key)
    print(("key_release", key))


# keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
# keyboard_listener.run()

keycode_ctrl = {
    "a": "\x01",
    "b": "\x02",
    "c": "\x03",
    "d": "\x04",
    "e": "\x05",
    "f": "\x06",
    "g": "\x07",
    "h": "\x08",
    "i": "\x09",
    "j": "\x0a",
    "k": "\x0b",
    "l": "\x0c",
    "m": "\x0d",
    "n": "\x0e",
    "o": "\x0f",
    "p": "\x10",
    "q": "\x11",
    "r": "\x12",
    "s": "\x13",
    "t": "\x14",
    "u": "\x15",
    "v": "\x16",
    "w": "\x17",
    "x": "\x18",
    "y": "\x19",
    "z": "\x1a",
    "[": "\x1b",
    "[": "<219>",
    "]": "\x1d",
    "]": "<221>",
    "-": "\x1f",
    "-": "<189>",
    "=": "<187>",
    "`": "<192>",
    "0": "<48>",
    "1": "<49>",
    "2": "<50>",
    "3": "<51>",
    "4": "<52>",
    "5": "<53>",
    "6": "<54>",
    "7": "<55>",
    "8": "<56>",
    "9": "<57>",
    "\\": "\x1c",
    "\\": "<220>",
    ";": "<186>",
    "'": "<222>",
    "<": "<188>",
    ">": "<190>",
    "?": "<191>",
}
print({v: k for k, v in keycode_ctrl.items()})
