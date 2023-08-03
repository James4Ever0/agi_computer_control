import sys

sys.path.append("../")

# TODO: add feedback to unsupported keys.
# for now, just ignore these. don't do anything.

from conscious_struct import HIDActionTypes
import json

trans_outpath = "translation_keys.json"
from functools import lru_cache


@lru_cache(maxsize=1)
def load_translation_table():
    with open(trans_outpath, "r") as f:
        content = f.read()
        data = json.loads(content)
        tk = "translation_table"
        data[tk] = {k: bytes.fromhex(v) for k, v in data[tk].items()}
        return data


from beartype import beartype


@beartype
def KeyLiteralToKCOMKeycode(keyLiteral: HIDActionTypes.keys):
    translation_data = load_translation_table()
    translation_table, missing = (
        translation_data["translation_table"],
        translation_data["missing"],
    )
    if keyLiteral in translation_table.keys():
        return translation_table[keyLiteral]

    elif keyLiteral in missing:
        raise Exception("Calling missing keyLiteral:", keyLiteral)
    else:
        raise Exception("Unknown keyLiteral: " + keyLiteral)

from beartype.door import is_bearable
if __name__ == "__main__":
    # import json5
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

        keyname = keyname.replace("Arrow", "").strip()

        base_trans0 = keyname.replace(" ", "_").lower()
        base_trans = (
            base_trans0.replace("gui", "cmd")
            .replace("control", "ctrl")
            .replace("escape", "esc")
            .replace("keyboard", "media")
            .replace("mute", "volume_mute")
            .replace("volume_dn", "volume_down")
            .replace("return", "enter")
        )
        missing_key_literals = [
        ]

        def check_is_common_keyname(keyname: str):
            if keyname.startswith("Key"):
                keyname = keyname.strip("Key.")
                for forbidden_prefix in ["ctrl", "cmd", "shift", "alt", "media"]:
                    if keyname.startswith(forbidden_prefix):
                        return False
            return True

        def do_append(t):
            _k = f"Key.{t}"
            if check_is_common_keyname(_k):
                possible_translations.append(t)
                possible_translations.append(_k)
            else:
                for k in [t, _k]:
                    if is_bearable(k, HIDActionTypes.keys):
                        missing_key_literals.append(k)

        do_append(base_trans)
        for direction in ["right_", "left_"]:
            if base_trans.startswith(direction):
                base_trans = base_trans.replace(direction, "") + f"_{direction[0]}"
                if direction == "left_":
                    basekey = base_trans.split("_")[0]
                    do_append(basekey)
                do_append(base_trans)
        if not base_trans0.startswith("F"):
            if len(keyname) == 3 and keyname[1] == " ":
                val = keyname[0]
                trans = f"""'{val}'""" if val != "'" else f'''"{val}"'''
                possible_translations.append(trans)

        for translation in possible_translations:
            kcom_translation_table[translation] = keycode

    # coverage test.
    error_msg = []


    translation_table_cleaned = {}
    import rich
    # extra_missing_key_literals = []

    for key_literal in HIDActionTypes.keys.__args__:
        if key_literal not in kcom_translation_table:
            # if key_literal not in missing_key_literals:
            if check_is_common_keyname(key_literal):
                error_msg.append(f"{key_literal} not covered by translation table.")
            else:
                missing_key_literals.append(key_literal)
        else:
            keycode = kcom_translation_table[key_literal]
            translation_table_cleaned.update({key_literal: keycode})
    if error_msg:
        raise Exception("\n" + "\n".join(error_msg))
    print("cleaned translation table:")
    rich.print(translation_table_cleaned)
    # use bytes.fromhex() to deserialize.
    output_data = {
        "translation_table": {k: v.hex() for k, v in translation_table_cleaned.items()},
        "missing": missing_key_literals
        # "missing": missing_key_literals + extra_missing_key_literals,
    }
    with open(trans_outpath, "w+") as f:
        content = json.dumps(output_data, indent=4)
        f.write(content)
    print("write to:", trans_outpath)
