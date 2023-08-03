import serial
from beartype import beartype
from beartype.vale import Is
from typing import Annotated
from typing_extensions import TypeAlias
# for branching; ref: https://beartype.readthedocs.io/en/latest/api_door/
from beartype.door import is_bearable
from enum import Enum, auto
from functools import reduce
from typing import Union, List, Literal

one_byte: TypeAlias = Annotated[bytes, Is[lambda b: len(b) == 1]]
two_bytes: TypeAlias = Annotated[bytes, Is[lambda b: len(b) == 2]]
four_bytes: TypeAlias = Annotated[bytes, Is[lambda b: len(b) == 4]]
six_bytes: TypeAlias = Annotated[bytes, Is[lambda b: len(b) == 6]]
eight_bytes: TypeAlias = Annotated[bytes, Is[lambda b: len(b) == 8]]

# confusing!
serialDevices = {
    "power": "/dev/serial/by-id/usb-1a86_5523-if00-port0",
    "hid": "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
    # another hid device will be: ch9392
}

deviceType = "power"
# deviceType = "hid"  # 为了保证数据能正常传输，两条数据发送间隔最低要有5ms 的延时；意思就是你发送一个数据后延时5ms 再发下一条数据。

ser = serial.Serial(serialDevices[deviceType], timeout=0.01,
                    **({'baudrate': 57600} if deviceType == 'hid' else {}))
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


if deviceType == "power":
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

elif deviceType == "hid":
    commonHeader = b"\x57\xab"

    class KCOMHeader(Enum):
        modifyIDHeader = commonHeader+b"\x10"  # +4bytes, (2bytes VID, 2bytes PID)
        keyboardHeader = commonHeader+b"\x01"  # +8bytes
        mouseRelativeHeader = commonHeader+b"\x02"  # +4bytes

        # below only working for KCOM3
        multimediaHeader = commonHeader+b"\x03"  # +(2 or 4)bytes
        mouseAbsoluteHeader = commonHeader+b"\x04"  # +4bytes

    @beartype
    def kcom_write_and_read(header:KCOMHeader, data_code:bytes, length: Union[int, None]):
        if length:
            assert (data_length:=len(data_code)) == length, f"Assumed data length: {data_length}\nActual length: {length}"
        write_and_read(header+data_code)

    @beartype
    def reduce_opcodes_to_bytes(opcodes:List[Union[one_byte, two_bytes]]):
        opcode = reduce(lambda x, y: x | y, opcodes)
        bytecode = opcode.to_bytes(1 if opcode <= 0xff else 2)
        return bytecode
    

    @beartype
    def changeID(vid: two_bytes, pid: two_bytes):
        print("change VID=%s, PID=%s" % (vid, pid))
        data_code = vid+pid
        kcom_write_and_read(KCOMHeader.modifyIDHeader, data_code, 4)

    # use int.to_bytes afterwards.
    class ControlCode(Enum):
        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return 2 ** (count)
        LEFT_CONTROL = auto()
        LEFT_SHIFT = auto()
        LEFT_ALT = auto()
        LEFT_GUI = auto()
        RIGHT_CONTROL = auto()
        RIGHT_SHIFT = auto()
        RIGHT_ALT = auto()
        RIGHT_GUI = auto()
    
    class KeyboardKey(Enum):
        ...

    @beartype
    def keyboard(control_codes: List[ControlCode], keycodes: Annotated[List[KeyboardKey], Is[lambda l: len(l) <= 6 and len(l) >= 0]]): # check for "HID Usage ID"
        reserved_byte = b"\x00"
        control_code = reduce_opcodes_to_bytes(control_codes)
        data_code = control_code + reserved_byte + b"".join(keycodes + ([b"\x00"]*(6-len(keycodes))))
        kcom_write_and_read(KCOMHeader.keyboardHeader, data_code, 8)

    class MouseButton(Enum):
        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return 2 ** (count)
        LEFT = auto()
        RIGHT = auto()
        MIDDLE = auto()

    @beartype
    def mouse_common(button_codes: List[MouseButton], x_code:Union[two_bytes, one_byte], y_code: Union[two_bytes, one_byte], scroll_code: one_byte, length: Literal[4, 6]):
        button_code = reduce_opcodes_to_bytes(button_codes)
        # button_opcode = reduce_opcodes(button_codes)
        # button_code = button_opcode.to_bytes()
        data_code = button_code + x_code+ y_code+ scroll_code # all 1byte
        kcom_write_and_read(KCOMHeader.mouseRelativeHeader, data_code, length)

    class MultimediaKey(Enum):
        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return 2 ** (count)
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

    assert len(MultimediaKey.__members__) == 3*8

    # @beartype
    # def multimedia_raw(data_code: Union[two_bytes, four_bytes]):
    
    @beartype
    def multimedia(multimedia_keys: List[MultimediaKey]):
        data_code = reduce_opcodes_to_bytes(multimedia_keys)
        # multimedia_opcode = reduce_opcodes(multimedia_keys)
        # data_code = multimedia_opcode.to_bytes(1 if multimedia_opcode <= 0xff else 2)
        # multimedia_raw(data_code)
        kcom_write_and_read(KCOMHeader.multimediaHeader, data_code)
    

else:
    raise Exception("Unknown device type: {deviceType}".format(
        deviceType=deviceType))

ser.close()
