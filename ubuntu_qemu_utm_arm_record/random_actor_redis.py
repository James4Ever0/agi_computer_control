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

# first let's use pyautogui as random actor.
# then may consider cross-platform RPA record/replay libs
# like: https://github.com/repeats/Repeat

# you may use pyinput or something else.

from functools import lru_cache
import random
import pyautogui

# there are several keys we should never touch.


KEY_CHARS = [
    "\t",
    "\n",
    "\r",
    " ",
    "!",
    '"',
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "<",
    "=",
    ">",
    "?",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
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
    "{",
    "|",
    "}",
    "~",
]

KEY_MOD = [
    "alt",
    "altleft",
    "altright",
    "shift",
    "shiftleft",
    "shiftright",
    "ctrl",
    "ctrlleft",
    "ctrlright",
]
KEY_WIN_MOD = [
    "win",
    "winleft",
    "winright",
]

KEY_MAC_MOD = [
    "option",
    "optionleft",
    "optionright",
    "command",
]

KEY_DIRECTION = [
    "down",
    "up",
    "right",
    "left",
]
KEY_SPECIAL = [
    "backspace",
    "capslock",
    "del",
    "delete",
    "tab",
    "home",
    "insert",
    "end",
    "enter",
    "esc",
    "escape",
    "pagedown",
    "pageup",
    "pgdn",
    "pgup",
    "return",
]

KEY_FUNC = [
    "fn",
    "f1",
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
    "f2",
    "f20",
    "f21",
    "f22",
    "f23",
    "f24",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
]

KEY_NUMPAD = [
    "num0",
    "num1",
    "num2",
    "num3",
    "num4",
    "num5",
    "num6",
    "num7",
    "num8",
    "num9",
    "numlock",
]

KEY_MORE = [
    "accept",
    "pause",
    "add",
    "apps",
    "browserback",
    "browserfavorites",
    "browserforward",
    "browserhome",
    "browserrefresh",
    "browsersearch",
    "browserstop",
    "clear",
    "convert",
    "decimal",
    "divide",
    "execute",
    "playpause",
    "prevtrack",
    "print",
    "printscreen",
    "prntscrn",
    "prtsc",
    "prtscr",
    "scrolllock",
    "select",
    "separator",
    "sleep",
    "space",
    "stop",
    "subtract",
    "volumedown",
    "volumemute",
    "volumeup",
    "yen",
    "final",
    "hanguel",
    "hangul",
    "hanja",
    "help",
    "junja",
    "kana",
    "kanji",
    "launchapp1",
    "launchapp2",
    "launchmail",
    "launchmediaselect",
    "modechange",
    "multiply",
    "nexttrack",
    "nonconvert",
]

ALL_KEYS = (
    KEY_CHARS
    + KEY_DIRECTION
    + KEY_MOD
    + KEY_MAC_MOD
    + KEY_WIN_MOD
    + KEY_SPECIAL
    + KEY_FUNC
    + KEY_NUMPAD
    + KEY_MORE
)


INIT_KEYS = KEY_CHARS + KEY_DIRECTION + KEY_MOD + KEY_WIN_MOD + KEY_SPECIAL


# turn off pyautogui failsafe.

pyautogui.FAILSAFE = False


def get_random_single_key():
    key = random.choice(INIT_KEYS)
    return key


def random_press_single_key():
    key = get_random_single_key()
    pyautogui.press(key)


# no keydown support? what about states?


def get_random_mod_key():
    key = random.choice(KEY_SPECIAL + KEY_WIN_MOD)
    return key


def random_mod_key_down():
    key = get_random_mod_key()
    try:
        pyautogui.keyDown(key)
    except:
        pass


def random_mod_key_up():
    key = get_random_mod_key()
    try:
        pyautogui.keyUp(key)
    except:
        pass


def get_random_offset():
    offset = random.randint(-100, 100)
    return offset


def random_mouse_move():
    xOffset = get_random_offset()
    yOffset = get_random_offset()
    pyautogui.move(xOffset, yOffset)


@lru_cache(maxsize=1)
def get_screen_size():
    return pyautogui.size()  # (width, height)


def get_random_screen_position():
    width, height = get_screen_size()
    return random.randint(0, width), random.randint(0, height)


def random_mouse_moveTo():
    x, y = get_random_screen_position()
    pyautogui.moveTo(x, y)


# mouse click, mouse move, mouse scroll, mouse double click


def random_mouse_scroll():
    # don't use hscroll/vscroll because it only supports linux
    pyautogui.scroll(get_random_offset())


MOUSE_BUTTONS = [pyautogui.LEFT, pyautogui.MIDDLE, pyautogui.RIGHT]


def get_random_mouse_button():
    button = random.choice(MOUSE_BUTTONS)
    return button


MOUSE_ACTIONS = [
    lambda: pyautogui.leftClick(),
    lambda: pyautogui.rightClick(),
    lambda: pyautogui.middleClick(),
    lambda: pyautogui.mouseDown(button=get_random_mouse_button()),
    lambda: pyautogui.mouseUp(button=get_random_mouse_button()),
]


def random_mouse_button_action():
    try:
        action = random.choice(MOUSE_ACTIONS)
    except:
        pass
    action()


if __name__ == "__main__":

    from utils import check_redis_on, check_redis_off, set_redis_off_on_exception

    set_redis_off_on_exception()

    if check_redis_on():
        try:
            random_keyboard_actions = [
                random_mod_key_down,
                random_mod_key_up,
                random_press_single_key,
            ]

            random_mouse_actions = [
                random_mouse_button_action,
                random_mouse_move,
                random_mouse_moveTo,
                random_mouse_scroll,
            ]

            random_actions = random_mouse_actions + random_keyboard_actions

            # for _ in range(10):
            while check_redis_off() is False:
                random_action = random.choice(random_actions)
                try:
                    random_action()
                except:
                    pass

            # to recover from mortality...
            # use try...finally or something...
            # context manager...
        finally:
            for key in KEY_MOD + KEY_WIN_MOD + KEY_MAC_MOD:
                try:
                    pyautogui.keyUp(key)
                except:
                    pass

            for button in MOUSE_BUTTONS:
                try:
                    pyautogui.mouseUp(button)
                except:
                    pass
