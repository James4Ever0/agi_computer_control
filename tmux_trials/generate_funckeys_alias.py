PREDEFINED_TEMPLATE = {
    "BSpace": ["BackSpace"],
    "BTab": ["BackTab"],
    "Insert": ["IC", "InsertCharacter"],
    "Delete": ["DC", "DeleteCharacter"],
}

# TODO: imitate `/usr/lib/command-not-found` and handle inaccurate keystrokes from agent

def generate_case_alias(key: str):
    return [key.lower(), key.upper()]
