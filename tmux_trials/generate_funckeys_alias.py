PREDEFINED_TEMPLATE = {
    "BSpace": ["BackSpace"],
    "BTab": ["BackTab"],
    "Insert": ["IC", "InsertCharacter"],
    "Delete": ["DC", "DeleteCharacter"],
}


def generate_case_alias(key: str):
    return [key.lower(), key.upper()]
