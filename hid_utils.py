from enum import Enum, auto, Flag

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
