import humps # pip install pyhumps

PREDEFINED_TEMPLATE = {
    "BSpace": ["BackSpace"],
    "BTab": ["BackTab"],
    "Ctrl": ["Control"],
    "Insert": ["IC", "InsertCharacter"],
    "Delete": ["DC", "DeleteCharacter"],
    "PgDn": ["PageDown", "NPage"],
    "PgUp": ["PageUp", "PPage"],
    "\\": ["BackSlash"],
    "[": ["LeftSquareBracket"],
    "]": ["RightSquareBracket"],
    "{": ["LeftCurlyBracket"],
    "}": ["RightCurlyBracket"],
    "(": ["LeftParenthesis"],
    ")": ["RightParenthesis"],
    "<": ["LessThan"],
    ">": ["GreaterThan"],
    "|": ["VerticalBar"],
    "-": ["Hyphen"],
    "+": ["Plus"],
    "=": ["Equals"],
    "*": ["Asterisk"],
    "/": ["Slash"],
    "%": ["Percent"],
    "#": ["Hash"],
    "&": ["Ampersand"],
    "$": ["Dollar"],
    "@": ["At"],
    "!": ["Exclamation"],
    "?": ["Question"],
    "^": ["Caret"],
    "~": ["Tilde"],
    "`": ["Backtick"],
    "'": ["SingleQuote"],
    '"': ["DoubleQuote"],
    ":": ["Colon"],
    ";": ["Semicolon"],
    ".": ["Period"],
    ",": ["Comma"],
    "_": ["Underscore"],
}

FUNCKEY_LETTERS = {"C": "Ctrl", "S": "Shift", "M": "Meta"}

KEYPAD_SHORTHAND = "KP"
KEYPAD_FULLNAME = "KeyPad"

# TODO: imitate `/usr/lib/command-not-found` and handle inaccurate keystrokes from agent

def generate_kebabize(key:str):
    humps.camelize("jack_in_the_box")  # jackInTheBox
    humps.decamelize("rubyTuesdays")  # ruby_tuesdays
    humps.pascalize("red_robin")  # RedRobin
    humps.kebabize("white_castle")  # white-castle

def generate_case_aliases(key: str):
    return key.lower(), key.upper(), key.title()


def generate_hotkey_with_different_connectors(hotkey: str):
    plus_connected_hotkey = hotkey.replace("-", "+")
    return plus_connected_hotkey

def generate_all_combination_hotkeys(key:str):
    ...

def test():
    test_cases = [
        (
            "C-Tab",
            [
                "Ctrl-Tab",
                "Control-Tab",
                "c-tab",
                "c+tab",
                "ctrl-tab",
                "ctrl+tab",
                "control-tab",
                "control+tab",
                "C-TAB",
                "C+TAB",
                "CTRL+TAB",
                "CTRL-TAB",
                "CONTROL+TAB",
                "CONTROL-TAB",
            ],
        ),
    ]


if __name__ == "__main__":
    test()
