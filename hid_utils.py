from enum import Enum, auto, Flag
from beartype.vale import Is
from typing_extensions import Annotated, TypeAlias
from conscious_struct import HIDActionTypes, HIDActionBase
from log_utils import logger_print


def length_limit(l):
    return Is[lambda b: len(b) == l]


# import Xlib
# python-xlib
import os

sourcefile_dirname = os.path.dirname(os.path.abspath(__file__))

key_literal_to_xk_keysym_translation_table_path = os.path.join(
    sourcefile_dirname, "KL2XKS.json"
)
from functools import lru_cache
import json


@lru_cache
def getKL2XKS():
    with open(key_literal_to_xk_keysym_translation_table_path, "r") as f:
        KL2XKS = json.loads(f.read())
    return KL2XKS


from beartype import beartype


@beartype
def strip_key_literal(key_literal: HIDActionTypes.keys):
    is_special, is_media = False, False
    keychar = ...
    if key_literal.startswith(prefix := "Key."):
        is_special = True
        keychar = key_literal.lstrip(prefix)
        if key_literal.startswith(prefix := "media_"):
            is_media = True
            keychar = keychar.lstrip(prefix)
    if len(key_literal) == 3:
        if key_literal[0] == key_literal[2] != (keychar := key_literal[1]):
            keychar = keychar
        else:
            raise Exception(f"Abnormal enclosed keychar: {repr(key_literal)}")
    if keychar == Ellipsis:
        raise Exception(f"Unable to strip key literal: {repr(key_literal)}")
    else:
        return is_special, is_media, keychar


@beartype
def key_literal_to_xk_keysym(key_literal: HIDActionTypes.keys):
    # is_special, is_media, stripped_key_literal = strip_key_literal(key_literal)
    KL2XKS = getKL2XKS()
    xk_keysym = KL2XKS.get(key_literal)
    return xk_keysym
    # Xlib.XK.string_to_keysym(stripped_key_literal)
    # generate this translation table statically, then we will review.


def byte_with_length_limit(l):
    return Annotated[bytes, length_limit(l)]


one_byte: TypeAlias = byte_with_length_limit(1)
two_bytes: TypeAlias = byte_with_length_limit(2)
four_bytes: TypeAlias = byte_with_length_limit(4)
six_bytes: TypeAlias = byte_with_length_limit(6)
eight_bytes: TypeAlias = byte_with_length_limit(8)

non_neg_int: TypeAlias = Annotated[int, Is[lambda i: i >= 0]]
pos_int: TypeAlias = Annotated[int, Is[lambda i: i > 0]]

movement: TypeAlias = Annotated[
    int, Is[lambda i: i >= -126 and i <= 126]
]  # this is hardware limit. software might not be limited. (shall we adapt to software limit instead of hardware)


class ControlCode(Flag):
    # @staticmethod
    # def _generate_next_value_(name, start, count, last_values):
    #     return 2 ** (count)
    NULL = 0

    LEFT_CONTROL = auto()
    LEFT_SHIFT = auto()
    LEFT_ALT = auto()
    LEFT_GUI = auto()

    RIGHT_CONTROL = auto()
    RIGHT_SHIFT = auto()
    RIGHT_ALT = auto()
    RIGHT_GUI = auto()


class MouseButton(Flag):
    # class MouseButton(Enum):
    # @staticmethod
    # def _generate_next_value_(name, start, count, last_values):
    #     return 2 ** (count)

    NULL = 0

    LEFT = auto()
    RIGHT = auto()
    MIDDLE = auto()


class MultimediaKey(Flag):
    # class MultimediaKey(Enum):
    # @staticmethod
    # def _generate_next_value_(name, start, count, last_values):
    #     return 2 ** (count)
    Null = 0

    # row 1
    VolumeUp = auto()
    VolumeDown = auto()
    Mute = auto()
    PlayPause = auto()
    NextTrack = auto()
    PreviousTrack = auto()
    CDStop = auto()
    Eject = auto()

    # row 2
    EMail = auto()
    WWWSearch = auto()
    WWWFavourites = auto()
    WWWHome = auto()
    WWWBack = auto()
    WWWForward = auto()
    WWWStop = auto()
    Refresh = auto()

    # row 3
    Media = auto()
    Explorer = auto()
    Calculator = auto()
    ScreenSave = auto()
    MyComputer = auto()
    Minimize = auto()
    Record = auto()
    Rewind = auto()


assert len(MultimediaKey.__members__) == 3 * 8 + 1  # include "Null"


class ACPIKey(Flag):
    Null = 0  # for clearing all "ACPI" keys.

    Power = auto()
    Sleep = auto()
    Wakeup = auto()


if __name__ == "__main__":
    # generate that table.
    import Levenshtein as L
    import keysymdef

    unicode_str_to_xk_keysym = {}
    xk_keysyms = []
    xk_keysyms_lut = {}

    for xk_keysym, _, unicode_int in keysymdef.keysymdef:
        unicode_str = None
        as_unicode_char = False
        if unicode_int:
            try:
                unicode_str = chr(unicode_int)
                unicode_str_to_xk_keysym[unicode_str] = xk_keysym
                as_unicode_char = True
            except:
                pass
        xk_keysym_lower = xk_keysym.lower()
        xk_keysyms_lut[xk_keysym_lower] = xk_keysym
        
        if not as_unicode_char:
            xk_keysyms.append(xk_keysym_lower)

    KL2XKS = {}
    # import rich

    # rich.print(xk_keysyms_lut)
    # breakpoint()
    keywords_translation_table = dict(
        cmd="super",
        ctrl="control",
        _left="_l",
        _right="_r",
        esc="escape",
        enter="return",
    )
    from typing import Dict

    def translate(string: str, translation_table: Dict[str, str]):
        for k, v in translation_table.items():
            string = string.replace(k, v)
        return string

    for key_literal in HIDActionBase.keys:
        is_special, is_media, stripped_key_literal = strip_key_literal(key_literal)
        # media prefix is removed.
        if stripped_key_literal in unicode_str_to_xk_keysym.keys():
            keysym = unicode_str_to_xk_keysym[stripped_key_literal]
        else:
            # import humps
            stripped_key_literal = translate(
                stripped_key_literal.lower(), keywords_translation_table
            )
            # if "return" in stripped_key_literal:
            #     breakpoint()
            xk_keysyms.sort(
                key=lambda keysym: L.distance(keysym.lower(), stripped_key_literal)
            )
            keysym = xk_keysyms.pop(0)
        KL2XKS[key_literal] = xk_keysyms_lut[keysym]
    with open(key_literal_to_xk_keysym_translation_table_path, "w+") as f:
        f.write(json.dumps(KL2XKS, ensure_ascii=False, indent=4))
    logger_print("write to:", key_literal_to_xk_keysym_translation_table_path)
