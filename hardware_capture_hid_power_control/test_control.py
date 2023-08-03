import serial
from beartype import beartype
from beartype.vale import Is
from typing import Annotated
from typing_extensions import TypeAlias
from enum import StrEnum

# for branching; ref: https://beartype.readthedocs.io/en/latest/api_door/
from beartype.door import is_bearable
from enum import Enum, auto, Flag
from functools import reduce
from typing import Union, List, Literal, Tuple
from common_keycodes import KeyLiteralToKCOMKeycode, HIDActionTypes

length_limit = lambda l: Is[lambda b: len(b) == l]
byte_with_length_limit = lambda l: Annotated[bytes, length_limit(l)]

one_byte: TypeAlias = byte_with_length_limit(1)
two_bytes: TypeAlias = byte_with_length_limit(2)
four_bytes: TypeAlias = byte_with_length_limit(4)
six_bytes: TypeAlias = byte_with_length_limit(6)
eight_bytes: TypeAlias = byte_with_length_limit(8)

non_neg_int: TypeAlias = Annotated[int, Is[lambda i: i >= 0]]
pos_int: TypeAlias = Annotated[int, Is[lambda i: i > 0]]

movement: TypeAlias = Annotated[int, Is[lambda i: i >= -126 and i <= 126]]

# confusing!

class DeviceType(StrEnum):
    power = auto()
    hid = auto()
    ch9329 = auto()


serialDevices = {
    DeviceType.power: "/dev/serial/by-id/usb-1a86_5523-if00-port0",
    DeviceType.hid: "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
    # another hid device will be: ch9329
    DeviceType.ch9329: ...,
}

deviceType = DeviceType.power
# deviceType = DeviceType.ch9329
# deviceType = DeviceType.hid  # 为了保证数据能正常传输，两条数据发送间隔最低要有5ms 的延时；意思就是你发送一个数据后延时5ms 再发下一条数据。

ser = serial.Serial(
    serialDevices[deviceType],
    timeout=0.01,
    **({"baudrate": 57600} if deviceType == "hid" else {}),
)
print("Serial device: %s" % deviceType)

# print(dir(ser))
# ['BAUDRATES', 'BAUDRATE_CONSTANTS', 'BYTESIZES', 'PARITIES', 'STOPBITS', '_SAVED_SETTINGS', ..., '_baudrate', '_break_state', '_bytesize', '_checkClosed', '_checkReadable', '_checkSeekable', '_checkWritable', '_dsrdtr', '_dtr_state', '_exclusive', '_inter_byte_timeout', '_parity', '_port', '_reconfigure_port', '_reset_input_buffer', '_rs485_mode', '_rts_state', '_rtscts', '_set_rs485_mode', '_set_special_baudrate', '_stopbits', '_timeout', '_update_break_state', '_update_dtr_state', '_update_rts_state', '_write_timeout', '_xonxoff', 'applySettingsDict', 'apply_settings', 'baudrate', 'break_condition', 'bytesize', 'cancel_read', 'cancel_write', 'cd', 'close', 'closed', 'cts', 'dsr', 'dsrdtr', 'dtr', 'exclusive', 'fd', 'fileno', 'flush', 'flushInput', 'flushOutput', 'getCD', 'getCTS', 'getDSR', 'getRI', 'getSettingsDict', 'get_settings', 'inWaiting', 'in_waiting', 'interCharTimeout', 'inter_byte_timeout', 'iread_until', 'isOpen', 'is_open', 'isatty', 'name', 'nonblocking', 'open', 'out_waiting', 'parity', 'pipe_abort_read_r', 'pipe_abort_read_w', 'pipe_abort_write_r', 'pipe_abort_write_w', 'port', 'portstr', 'read', 'read_all', 'read_until', 'readable', 'readall', 'readinto', 'readline', 'readlines', 'reset_input_buffer', 'reset_output_buffer', 'ri', 'rs485_mode', 'rts', 'rtscts', 'seek', 'seekable', 'sendBreak', 'send_break', 'setDTR', 'setPort', 'setRTS', 'set_input_flow_control', 'set_low_latency_mode', 'set_output_flow_control', 'stopbits', 'tell', 'timeout', 'truncate', 'writable', 'write', 'writeTimeout', 'write_timeout', 'writelines', 'xonxoff']

# import rich
# rich.print(ser.__dict__)]

# print(ser.name) # /dev/serial/by-id/usb-1a86_5523-if00-port0

# ser.write(b"hello")


@beartype
def write_and_read(_bytes: bytes):
    ser.write(_bytes)
    print(f"w> {repr(_bytes)}")
    res = ser.readall()
    print(f"r> {repr(res)}")

# cannot use match here? python 3.10+ required

if deviceType == DeviceType.power:
    # will reset on reboot
    channel = 1  # CH3 does not exist. CH2 is placeholder. (virtually working)
    # channel = 2

    # state = "ON"
    # # state = "OFF"

    # write_and_read(f"CH{channel}=?".encode())
    # write_and_read(f"CH{channel}={state}".encode())
    # write_and_read(f"CH{channel}=?".encode())

    # just toggle.
    write_and_read(f"CH{channel}=OFF".encode())
    write_and_read(f"CH{channel}=ON".encode())

elif deviceType == DeviceType.hid:
    commonHeader = b"\x57\xab"

    class KCOMHeader(Enum):
        modifyIDHeader = commonHeader + b"\x10"  # +4bytes, (2bytes VID, 2bytes PID)
        keyboardHeader = commonHeader + b"\x01"  # +8bytes
        mouseRelativeHeader = commonHeader + b"\x02"  # +4bytes

        # below only working for KCOM3
        multimediaHeader = commonHeader + b"\x03"  # +(2 or 4)bytes
        mouseAbsoluteHeader = commonHeader + b"\x04"  # +4bytes

    @beartype
    def kcom_write_and_read(
        header: KCOMHeader, data_code: bytes, length: Union[int, List[int], None]
    ):
        if is_bearable(length, int):
            length = [length]
        if length:
            assert (
                data_length := len(data_code)
            ) == length, f"Assumed data lengths: {length}\nActual length: {data_length}"
        write_and_read(header + data_code)

    import math

    @beartype
    def reduce_flags_to_bytes(
        flags: List[Flag],
        byteorder: Literal["little", "big"] = "little",
        byte_length: Union[int, Ellipsis] = ...,
    ):
        # def reduce_flags_to_bytes(opcodes: List[Union[one_byte, two_bytes]]):
        flag = reduce(lambda a, b: a | b, flags)
        opcode = flag.value

        # bytecode = opcode.to_bytes(1 if opcode <= 0xFF else 2)
        if byte_length is ...:
            byte_length = (
                get_byte_length := lambda _bytes: math.ceil(
                    len(hex(_bytes).strip("0x")) / 2
                )
            )(opcode)
            for member in type(flags[0]).__members__.values():
                if (member_byte_length := get_byte_length(member.value)) > byte_length:
                    byte_length = member_byte_length

        byte_code = opcode.to_bytes(byte_length, byteorder=byteorder)

        return byte_code

    @beartype
    def changeID(vid: two_bytes, pid: two_bytes):
        print("change VID=%s, PID=%s" % (vid, pid))
        data_code = vid + pid
        kcom_write_and_read(KCOMHeader.modifyIDHeader, data_code, 4)

    # use int.to_bytes afterwards.
    # use enum.Flag to replace enum.Enum in this situation.
    class ControlCode(Flag):
        # @staticmethod
        # def _generate_next_value_(name, start, count, last_values):
        #     return 2 ** (count)

        LEFT_CONTROL = auto()
        LEFT_SHIFT = auto()
        LEFT_ALT = auto()
        LEFT_GUI = auto()
        RIGHT_CONTROL = auto()
        RIGHT_SHIFT = auto()
        RIGHT_ALT = auto()
        RIGHT_GUI = auto()

    # class KeyboardKey(Enum):
    #     ...

    @beartype
    def keyboard(
        control_codes: List[ControlCode],
        key_literals: Annotated[
            List[HIDActionTypes.keys], Is[lambda l: len(l) <= 6 and len(l) >= 0]
        ],
    ):  # check for "HID Usage ID"
        reserved_byte = b"\x00"
        control_code = reduce_flags_to_bytes(control_codes)
        keycodes = [
            KeyLiteralToKCOMKeycode(key_literal)
            for key_literal in key_literals
            if KeyLiteralToKCOMKeycode(key_literal)
        ]  # could reduce size with walrus operator with higher python version.
        # keycodes = [v:=KeyLiteralToKCOMKeycode(key_literal) for key_literal in key_literals if v]
        data_code = (
            control_code
            + reserved_byte
            + b"".join(keycodes + ([b"\x00"] * (6 - len(keycodes))))
        )
        kcom_write_and_read(KCOMHeader.keyboardHeader, data_code, 8)

    class MouseButton(Flag):
        # class MouseButton(Enum):
        # @staticmethod
        # def _generate_next_value_(name, start, count, last_values):
        #     return 2 ** (count)

        LEFT = auto()
        RIGHT = auto()
        MIDDLE = auto()

    @beartype
    def get_rel_code(c_rel: movement):
        if c_rel < 0:
            c_rel = -c_rel + 0x80
        return c_rel.to_bytes()

    @beartype
    def mouse_common(
        button_codes: List[MouseButton],
        x_code: Union[two_bytes, one_byte],
        y_code: Union[two_bytes, one_byte],
        scroll: movement,
        kcom_flag: Literal[
            KCOMHeader.mouseRelativeHeader, KCOMHeader.mouseAbsoluteHeader
        ],
    ):
        scroll_code = get_rel_code(scroll)
        button_code = reduce_flags_to_bytes(button_codes)
        # button_opcode = reduce_opcodes(button_codes)
        # button_code = button_opcode.to_bytes()
        data_code = button_code + x_code + y_code + scroll_code  # all 1byte
        kcom_write_and_read(
            kcom_flag,
            data_code,
            4 if is_bearable(kcom_flag, KCOMHeader.mouseRelativeHeader) else 6,
        )

    @beartype
    def mouse_relative(
        button_codes: List[MouseButton], x: movement, y: movement, scroll: movement
    ):
        x_code = get_rel_code(x)
        y_code = get_rel_code(y)

        mouse_common(
            button_codes,
            x_code,
            y_code,
            scroll,
            kcom_flag=KCOMHeader.mouseRelativeHeader,
        )

    @beartype
    def mouse_absolute(
        button_codes: List[MouseButton],
        coordinate: Tuple[non_neg_int, non_neg_int],
        resolution: Tuple[pos_int, pos_int],
        scroll: movement,
    ):
        """
        coordinate: (x_abs, y_abs)
        resolution: (width, height)
        """

        (x_abs, y_abs) = coordinate
        (width, height) = resolution

        assert x_abs <= width, f"Invalid x: {x_abs}\nWidth: {width}"
        assert y_abs <= height, f"Invalid y: {y_abs}\nHeight: {height}"

        get_abs_code = lambda c_abs, res: int((4096 * c_abs) / res).to_bytes(
            2, byteorder="little"
        )

        x_code = get_abs_code(x_abs)
        y_code = get_abs_code(y_abs)

        # scroll_code = get_rel_code(scroll)

        mouse_common(
            button_codes,
            x_code,
            y_code,
            scroll,
            kcom_flag=KCOMHeader.mouseAbsoluteHeader,
        )

    class MultimediaKey(Flag):
        # class MultimediaKey(Enum):
        # @staticmethod
        # def _generate_next_value_(name, start, count, last_values):
        #     return 2 ** (count)

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

    assert len(MultimediaKey.__members__) == 3 * 8

    class ACPIKey(Flag):
        Power = auto()
        Sleep = auto()
        Wakeup = auto()

    # @beartype
    # def multimedia_raw(data_code: Union[two_bytes, four_bytes]):

    @beartype
    def multimedia(keys: Union[List[ACPIKey], List[MultimediaKey]]):
        isMultimediaKeys = is_bearable(keys, List[MultimediaKey])
        key_code = reduce_flags_to_bytes(keys)
        data_code = (b"\x02" if isMultimediaKeys else b"\x01") + key_code
        # multimedia_opcode = reduce_opcodes(multimedia_keys)
        # data_code = multimedia_opcode.to_bytes(1 if multimedia_opcode <= 0xff else 2)
        # multimedia_raw(data_code)
        kcom_write_and_read(
            KCOMHeader.multimediaHeader, data_code, 6 + (2 if isMultimediaKeys else 0)
        )

elif deviceType == DeviceType.ch9329:
    ...
else:
    raise Exception("Unknown device type: {deviceType}".format(deviceType=deviceType))

ser.close()
