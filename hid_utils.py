from enum import Enum, auto, Flag
from beartype.vale import Is
from typing_extensions import Annotated, TypeAlias
from conscious_struct import HIDActionTypes


def length_limit(l):
    return Is[lambda b: len(b) == l]

# import Xlib
import keysymdef

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
    if keychar is Ellipsis:
        raise Exception(f"Unable to strip key literal: {repr(key_literal)}")
    else:
        return is_special, is_media, keychar

@beartype
def key_literal_to_xk_keysym(key_literal: HIDActionTypes.keys):
    is_special, is_media, stripped_key_literal = strip_key_literal(key_literal)
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
