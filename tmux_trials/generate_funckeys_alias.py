import humps
from itertools import permutations
from lib import json_pretty_print

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

PREDEFINED_ALIASES = {
    "BSpace": ["BackSpace"],
    "BTab": ["BackTab"],
    "Ctrl": ["Control"],
    "Esc": ["Escape"],
    "Insert": ["IC", "InsertCharacter"],
    "Delete": ["DC", "DeleteCharacter"],
    "PgDn": ["PageDown", "NPage"],
    "PgUp": ["PageUp", "PPage"],
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
    "Up": ["ArrowUp","CursorUp", ],
    "Down": ["ArrowDown","CursorDown", ],
    "Left": ["ArrowLeft","CursorLeft", ],
    "Down": ["ArrowDown","CursorDown", ],
    "Meta": ["Alt"],
}

PREDEFINED_ALIASES_REVERSE_MAP = {
    it:k for k,v in PREDEFINED_ALIASES.items() for it in v
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
    plus_connected_hotkey = hotkey.replace("-", "+")
    return plus_connected_hotkey


def generate_all_permutation_hotkeys(key: str):
    components = key.split("-")
    ret = []
    prefixes = components[:-1]
    suffix = components[-1]
    for it in permutations(prefixes):
        ret.append("-".join(it + [suffix]))
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


def generate_funckeys_aliases():
    ret = {}
    for key in FUNC_KEYS:
        aliases = set()
        fullkey = key.replace("F", "FuncKey")
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

def generate_display_and_update_aliases(candidates:dict, name:str, ret:dict):
    print(f"[*] {name} aliases:")
    json_pretty_print(candidates)
    ret.update(candidates)

def generate_all_aliases():
    ret = {}
    generate_display_and_update_aliases(generate_funckeys_aliases(),"Funckey",ret)
    generate_display_and_update_aliases(generate_additionalkey_aliases(),"Additionalkey",ret)
    
    return ret


def main():
    all_aliases = generate_all_aliases()


if __name__ == "__main__":
    # test()
    main()
