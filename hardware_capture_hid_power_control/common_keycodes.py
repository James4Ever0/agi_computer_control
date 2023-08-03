import sys

sys.path.append("../")

from conscious_struct import HIDActionTypes
import json

with open("keys.json", "r") as f:
    content = f.read()
    kcom_keycodes = json.loads(content)

kcom_translation_table = {}

import re


def subs_brackets(e):
    s, _ = re.subn(r"\(.*\)", "", e)
    s = s.strip()
    return s


for record in kcom_keycodes:
    keyname = subs_brackets(record["Key Name"])
    keycode = bytes.fromhex(record["HID Usage ID"])

    possible_translations = []

    base_trans0 = keyname.replace(" ", "_").lower()
    base_trans = base_trans0.replace("gui", "cmd")
    possible_translations.append(base_trans)
    possible_translations.append(f"Key.{base_trans}")
    for direction in ["right_", "left_"]:
        if base_trans.startswith(direction):
            base_trans = base_trans.replace(direction,"") + f"_{direction[0]}"
            possible_translations.append(base_trans)
    if (
        not base_trans0.startswith("F")
    ):
        if (len(keyname) == 3
        and keyname[1] == " "
    ):
            val = keyname[0]
            trans = f"""'{val}'""" if val != "'" else f'''"{val}"'''
            possible_translations.append(trans)


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
