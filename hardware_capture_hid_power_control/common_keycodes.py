import sys

sys.append("../")

from conscious_struct import HIDActionTypes
import json

with open("keys.json", "r") as f:
    content = f.read()
    kcom_keycodes = json.loads(content)

kcom_translation_table = {}

import re


def subs_brackets(e):
    s = re.subn(r"\(.*\)", "", e).strip()
    return s


for record in kcom_keycodes:
    keyname = subs_brackets(record["Key Name"])
    keycode = bytes.fromhex(record["HID Usage ID"])

    possible_translations = []

    possible_translations.append(keyname.replace(" ",lower())

    for translation in possible_translations:
        kcom_translation_table[translation] = keycode


def KeyLiteralToKCOMKeycode(keyLiteral: HIDActionTypes.keys):
    return kcom_translation_table(keyLiteral)


if __name__ == "__main__":
    # coverage test.
    error_msg = []
    for key_literal in HIDActionTypes.keys.__args__:
        if key_literal not in kcom_translation_table:
            error_msg.append(f"{key_literal} not covered by translation table.")
    if error_msg:
        raise Exception("\n".join(error_msg))
