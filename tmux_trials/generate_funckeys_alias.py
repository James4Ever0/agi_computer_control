import humps
from itertools import permutations
from lib import json_pretty_print
import json

# libraries providing naming style conversion
# dedicated: pip install pyhumps
# included: pip install flashback (lazero like utils inside)

# TODO: record commit activity everyday, count for work time. LOC. "git dashboard"

# TODO: integrate input-method like sequence inference capability to agent execution interface
# TODO: use json schema constrained or GBNF constrained grammer to restrict agent generation
# TODO: stepwise decoding in ollama: https://github.com/ollama/ollama/issues/6302

FUNC_KEYS = [
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
]

EXTRA_KEYS_WITH_ARROW_KEYS = [
    "IC",
    "Insert",
    "DC",
    "Delete",
    "Home",
    "End",
    "NPage",
    "PageDown",
    "PgDn",
    "PPage",
    "PageUp",
    "PgUp",
    "Tab",
    "BTab",
    "Space",
    "BSpace",
    "Enter",
    "Escape",
    "Up",
    "Down",
    "Left",
    "Right",
]

CTRL_HOTKEYS = [
    "C-Tab" "C-Space",
    "C-a",
    "C-b",
    "C-c",
    "C-d",
    "C-e",
    "C-f",
    "C-g",
    "C-h",
    "C-i",
    "C-j",
    "C-k",
    "C-l",
    "C-m",
    "C-n",
    "C-o",
    "C-p",
    "C-q",
    "C-r",
    "C-s",
    "C-t",
    "C-u",
    "C-v",
    "C-w",
    "C-x",
    "C-y",
    "C-z",
    "C-\\",
    "C-]",
    "C-^",
    "C-_",
]

CTRL_SHIFT_HOTKEYS = ["C-S-Tab"]

META_HOTKEYS = [
    "M-Escape",
    "M-Space",
    "M-!",
    'M-"',
    "M-#",
    "M-$",
    "M-%",
    "M-&",
    "M-'",
    "M-(",
    "M-)",
    "M-*",
    "M-+",
    "M-,",
    "M--",
    "M-.",
    "M-/",
    "M-0",
    "M-1",
    "M-2",
    "M-3",
    "M-4",
    "M-5",
    "M-6",
    "M-7",
    "M-8",
    "M-9",
    "M-:",
    "M-;",
    "M-<",
    "M-=",
    "M->",
    "M-?",
    "M-@",
    "M-A",
    "M-B",
    "M-C",
    "M-D",
    "M-E",
    "M-F",
    "M-G",
    "M-H",
    "M-I",
    "M-J",
    "M-K",
    "M-L",
    "M-M",
    "M-N",
    "M-O",
    "M-P",
    "M-Q",
    "M-R",
    "M-S",
    "M-T",
    "M-U",
    "M-V",
    "M-W",
    "M-X",
    "M-Y",
    "M-Z",
    "M-[",
    "M-\\",
    "M-]",
    "M-^",
    "M-_",
    "M-`",
    "M-a",
    "M-b",
    "M-c",
    "M-d",
    "M-e",
    "M-f",
    "M-g",
    "M-h",
    "M-i",
    "M-j",
    "M-k",
    "M-l",
    "M-m",
    "M-n",
    "M-o",
    "M-p",
    "M-q",
    "M-r",
    "M-s",
    "M-t",
    "M-u",
    "M-v",
    "M-w",
    "M-x",
    "M-y",
    "M-z",
    "M-{",
    "M-|",
    "M-}",
    "M-~",
    "M-Tab",
    "M-BSpace",
    "M-KP*",
    "M-KP+",
    "M-KP-",
    "M-KP.",
    "M-KP/",
    "M-KP0",
    "M-KP1",
    "M-KP2",
    "M-KP3",
    "M-KP4",
    "M-KP5",
    "M-KP6",
    "M-KP7",
    "M-KP8",
    "M-KP9",
]

KEYPAD_KEYS = [
    "KP*",
    "KP+",
    "KP-",
    "KP.",
    "KP/",
    "KP0",
    "KP1",
    "KP2",
    "KP3",
    "KP4",
    "KP5",
    "KP6",
    "KP7",
    "KP8",
    "KP9",
]

META_CONTROL_HOTKEYS = [
    "M-C-a",
    "M-C-b",
    "M-C-c",
    "M-C-d",
    "M-C-e",
    "M-C-f",
    "M-C-g",
    "M-C-h",
    "M-C-i",
    "M-C-j",
    "M-C-k",
    "M-C-l",
    "M-C-m",
    "M-C-n",
    "M-C-o",
    "M-C-p",
    "M-C-q",
    "M-C-r",
    "M-C-s",
    "M-C-t",
    "M-C-u",
    "M-C-v",
    "M-C-w",
    "M-C-x",
    "M-C-y",
    "M-C-z",
    "M-C-\\",
    "M-C-]",
    "M-C-^",
    "M-C-_",
]

EXTENDED_KEY_PREFIXES = [
    "S",
    "M",
    "S-M",
    "C",
    "S-C",
    "C-M",
    "S-C-M",
]

EXTENDED_KEY_SUFFIXES = [
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
    "Up",
    "Down",
    "Right",
    "Left",
    "Home",
    "End",
    "PPage",
    "PageUp",
    "PgUp",
    "NPage",
    "PageDown",
    "PgDn",
    "IC",
    "Insert",
    "DC",
    "Delete",
]


PREDEFINED_ALIASES = {
    "BSpace": ["BackSpace"],
    "BTab": ["BackTab"],
    "Ctrl": ["Control"],
    "Esc": ["Escape"],
    "Insert": ["IC", "InsertCharacter"],
    "Delete": ["DC", "DeleteCharacter"],
    "PgDn": ["PageDown", "NPage"],
    "PgUp": ["PageUp", "PPage"],
    "Enter": ["Return", "CarriageReturn", "CR", "LineFeed", "LF", "CRLF"],
    "Up": [
        "ArrowUp",
        "CursorUp",
    ],
    "Down": [
        "ArrowDown",
        "CursorDown",
    ],
    "Left": [
        "ArrowLeft",
        "CursorLeft",
    ],
    "Right": [
        "ArrowRight",
        "CursorRight",
    ],
    "Meta": ["Alt"],
    "\\": ["BackSlash"],
    "[": ["LeftSquareBracket"],
    "]": ["RightSquareBracket"],
    "{": ["LeftCurlyBracket"],
    "}": ["RightCurlyBracket"],
    "(": ["LeftParenthesis", "LP"],
    ")": ["RightParenthesis", "RP"],
    "<": ["LessThan", "Lt"],
    ">": ["GreaterThan", "Gt"],
    "|": ["VerticalBar", "VB"],
    "-": ["Hyphen"],
    "+": ["Plus"],
    "=": ["Equals", "Eq"],
    "*": ["Asterisk"],
    "/": ["Slash"],
    "%": ["Percent"],
    "#": ["Hash"],
    "&": ["Ampersand", "Amp"],
    "$": ["Dollar"],
    "@": ["At"],
    "!": ["Exclamation"],
    "?": ["Question"],
    "^": ["Caret"],
    "~": ["Tilde"],
    "`": ["Backtick"],
    "'": ["SingleQuote", "SQ"],
    '"': ["DoubleQuote", "DQ"],
    ":": ["Colon"],
    ";": ["Semicolon"],
    ".": ["Period"],
    ",": ["Comma"],
    "_": ["Underscore"],
    "0": ["DigitZero", "Zero"],
    "1": ["DigitOne", "One"],
    "2": ["DigitTwo", "Two"],
    "3": ["DigitThree", "Three"],
    "4": ["DigitFour", "Four"],
    "5": ["DigitFive", "Five"],
    "6": ["DigitSix", "Six"],
    "7": ["DigitSeven", "Seven"],
    "8": ["DigitEight", "Eight"],
    "9": ["DigitNine", "Nine"],
}

PREDEFINED_ALIASES_REVERSE_MAP = {
    it: k for k, v in PREDEFINED_ALIASES.items() for it in v
}

FUNCKEY_COMBOS = ("S-M", "S-C", "C-M", "S-C-M")
FUNCKEY_LETTERS = {"C": "Ctrl", "S": "Shift", "M": "Meta"}

KEYPAD_SHORTHAND = "KP"
KEYPAD_FULLNAME = "KeyPad"


# TODO: imitate `/usr/lib/command-not-found` and handle inaccurate keystrokes from agent


def generate_multiforms(key: str):
    baseform = humps.depascalize(key)
    camelform = humps.camelize(baseform)
    kebaform = humps.kebabize(baseform)
    return baseform, camelform, kebaform


def generate_case_aliases(key: str):
    return key.lower(), key.upper(), key.title()


def generate_hotkey_with_different_connectors(hotkey: str):
    ret = hotkey
    for it in FUNCKEY_LETTERS.keys():
        ret = ret.replace(f"{it}-", f"{it}+")
    return ret


def generate_all_permutation_hotkeys(key: str):
    components = key.split("-")
    ret = []
    prefixes = components[:-1]
    suffix = components[-1]
    for it in permutations(prefixes):
        ret.append("-".join(list(it) + [suffix]))
    return ret


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
    aliases = generate_funckeys_aliases()
    for key, value in test_cases:
        key_aliases = aliases[key]
        for it in value:
            assert it in key_aliases, f"Alias '{it}' not found for key '{key}'"
        remained_aliases = [it for it in key_aliases if it not in value]
        assert remained_aliases == [], f"Containing unexpected aliases for key '{key}': {remained_aliases}"

def generate_funckeys_aliases():
    ret = {}
    for key in FUNC_KEYS:
        aliases = set()
        fullkey_list = [
            key.replace("F", "FuncKey"),
            key.replace("F", "FunctionKey"),
            key.replace("F", "Function"),
            key.replace("F", "Func"),
            key.replace("F", "Fn"),
        ]
        for fullkey in fullkey_list:
            baseform, camelform, kebaform = generate_multiforms(fullkey)
            candidates = [key.lower(), fullkey, baseform, camelform, kebaform]
            aliases.update(candidates)
            for it in candidates:
                case_aliases = generate_case_aliases(it)
                aliases.update(case_aliases)
        ret[key] = list(aliases)
    return ret


def generate_additionalkey_aliases():
    ret = {}
    for key in EXTRA_KEYS_WITH_ARROW_KEYS:
        candidates = [key]
        aliases = set()
        if key in PREDEFINED_ALIASES_REVERSE_MAP.keys():
            continue
        if key in PREDEFINED_ALIASES.keys():
            candidates.extend(PREDEFINED_ALIASES[key])
        for it in candidates:
            baseform, camelform, kebaform = generate_multiforms(it)
            all_forms = [it, baseform, camelform, kebaform]
            aliases.update(all_forms)
            for form in all_forms:
                aliases.update(generate_case_aliases(form))
        ret[key] = list(aliases)
    return ret


def generate_display_and_update_aliases(candidates: dict, name: str, ret: dict):
    print(f"[*] {name} aliases:")
    json_pretty_print(candidates)
    ret.update(candidates)


def generate_all_aliases():
    ret = {}
    generate_display_and_update_aliases(generate_funckeys_aliases(), "Funckey", ret)
    generate_display_and_update_aliases(
        generate_additionalkey_aliases(), "Additionalkey", ret
    )

    return ret


def main():
    all_aliases = generate_all_aliases()
    key_aliases_lut = {it: k for k, v in all_aliases.items() for it in v}
    output_path = "key_aliases_lut.json"
    with open(output_path, "w+") as f:
        f.write(json.dumps(key_aliases_lut, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    # test()
    # main()
    generate_all_aliases()
