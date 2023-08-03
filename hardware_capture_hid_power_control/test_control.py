import math
import serial
from beartype import beartype
from beartype.vale import Is
from typing import Annotated
from typing_extensions import TypeAlias
from enum import StrEnum

# for branching; ref: https://beartype.readthedocs.io/en/latest/api_door/
from beartype.door import is_bearable
from enum import Enum, auto, Flag
import time
import random
from functools import reduce
from typing import Union, List, Literal, Tuple
from common_keycodes import KeyLiteralToKCOMKeycode, HIDActionTypes


def length_limit(l): return Is[lambda b: len(b) == l]
def byte_with_length_limit(l): return Annotated[bytes, length_limit(l)]


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


class MouseButton(Flag):
    # class MouseButton(Enum):
    # @staticmethod
    # def _generate_next_value_(name, start, count, last_values):
    #     return 2 ** (count)

    LEFT = auto()
    RIGHT = auto()
    MIDDLE = auto()


@beartype
def reduce_flags_to_bytes(
    flags: List[Flag],
    byteorder: Literal["little", "big"] = "little",
    byte_length: Union[int, Ellipsis] = ...,
):
    # def reduce_flags_to_bytes(opcodes: List[Union[one_byte, two_bytes]]):
    if flags == []:
        assert is_bearable(
            int, byte_length), f"invalid byte_length: {byte_length}"
        return b"\x00" * byte_length
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
        # +4bytes, (2bytes VID, 2bytes PID)
        modifyIDHeader = commonHeader + b"\x10"
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
        if length is not None:
            assert (
                data_length := len(data_code)
            ) == length, f"Assumed data lengths: {length}\nActual length: {data_length}"
        write_and_read(header + data_code)

    @beartype
    def changeID(vid: two_bytes, pid: two_bytes):
        print("change VID=%s, PID=%s" % (vid, pid))
        data_code = vid + pid
        kcom_write_and_read(KCOMHeader.modifyIDHeader, data_code, 4)

    # class KeyboardKey(Enum):
    #     ...

    # leave it empty to release all keys.
    @beartype
    def keyboard(
        control_codes: List[ControlCode] = [],
        key_literals: Annotated[
            List[HIDActionTypes.keys], Is[lambda l: len(
                l) <= 6 and len(l) >= 0]
        ] = [],
    ):  # check for "HID Usage ID"
        reserved_byte = b"\x00"
        control_code = reduce_flags_to_bytes(control_codes, byte_length=1)
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

    @beartype
    def get_rel_code(c_rel: movement):
        if c_rel < 0:
            c_rel = -c_rel + 0x80
        return c_rel.to_bytes()

    @beartype
    def mouse_common(
        x_code: Union[two_bytes, one_byte],
        y_code: Union[two_bytes, one_byte],
        scroll: movement,
        kcom_flag: Literal[
            KCOMHeader.mouseRelativeHeader, KCOMHeader.mouseAbsoluteHeader
        ],
        button_codes: List[MouseButton] = [],
    ):
        scroll_code = get_rel_code(scroll)
        button_code = reduce_flags_to_bytes(button_codes, byte_length=1)
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
        x: movement, y: movement, scroll: movement, button_codes: List[MouseButton] = []
    ):
        x_code = get_rel_code(x)
        y_code = get_rel_code(y)

        mouse_common(
            x_code,
            y_code,
            scroll,
            kcom_flag=KCOMHeader.mouseRelativeHeader,
            button_codes=button_codes,
        )

    @beartype
    def mouse_absolute(
        coordinate: Tuple[non_neg_int, non_neg_int],
        resolution: Tuple[pos_int, pos_int],
        scroll: movement,
        button_codes: List[MouseButton] = [],
    ):
        """
        coordinate: (x_abs, y_abs)
        resolution: (width, height)
        """

        (x_abs, y_abs) = coordinate
        (width, height) = resolution

        assert x_abs <= width, f"Invalid x: {x_abs}\nWidth: {width}"
        assert y_abs <= height, f"Invalid y: {y_abs}\nHeight: {height}"

        def get_abs_code(c_abs, res): return int((4096 * c_abs) / res).to_bytes(
            2, byteorder="little"
        )

        x_code = get_abs_code(x_abs)
        y_code = get_abs_code(y_abs)

        # scroll_code = get_rel_code(scroll)

        mouse_common(
            x_code,
            y_code,
            scroll,
            kcom_flag=KCOMHeader.mouseAbsoluteHeader,
            button_codes=button_codes,
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
    def multimedia(keys: Union[List[ACPIKey], List[MultimediaKey]] = []):
        isMultimediaKeys = is_bearable(keys, List[MultimediaKey])
        byte_length = 3 if isMultimediaKeys else 1
        key_code = reduce_flags_to_bytes(keys, byte_length=byte_length)
        data_code = (b"\x02" if isMultimediaKeys else b"\x01") + key_code
        # multimedia_opcode = reduce_opcodes(multimedia_keys)
        # data_code = multimedia_opcode.to_bytes(1 if multimedia_opcode <= 0xff else 2)
        # multimedia_raw(data_code)
        kcom_write_and_read(
            KCOMHeader.multimediaHeader, data_code, 4 + (1 + byte_length)
        )

elif deviceType == DeviceType.ch9329:
    import ch9329Comm

    # ref: https://github.com/beijixiaohu/CH9329_COMM
    class Keyboard:
        def __init__(
            self,
            port: serial.Serial,
        ):
            self.port = port

        @beartype
        def send_data(
            self,
            control_codes: List[ControlCode] = [],
            key_literals: Annotated[
                List[HIDActionTypes.keys], Is[lambda l: len(
                    l) <= 8 and len(l) >= 0]
            ] = [],
        ):
            # 将字符转写为数据包
            HEAD = b"\x57\xAB"  # 帧头
            ADDR = b"\x00"  # 地址
            CMD = b"\x02"  # 命令
            LEN = b"\x08"  # 数据长度
            DATA = b""  # 数据

            # 控制键
            control_byte = reduce_flags_to_bytes(control_codes, byte_length=1)
            DATA += control_byte
            # if ctrl == '':
            #     DATA += b'\x00'
            # elif isinstance(ctrl, int):
            #     DATA += bytes([ctrl])
            # else:
            #     DATA += self.control_button_hex_dict[ctrl]

            # DATA固定码
            DATA += b"\x00"

            # 读入data
            # for i in range(0, len(data), 2):
            #     DATA += self.normal_button_hex_dict[data[i:i + 2]]
            for key_literal in key_literals:
                DATA += KeyLiteralToKCOMKeycode(key_literal)
            if len(DATA) < 8:
                DATA += b"\x00" * (8 - len(DATA))
            else:
                DATA = DATA[:8]

            # 分离HEAD中的值，并计算和
            HEAD_hex_list = []
            for byte in HEAD:
                HEAD_hex_list.append(byte)
            HEAD_add_hex_list = sum(HEAD_hex_list)

            # 分离DATA中的值，并计算和
            DATA_hex_list = []
            for byte in DATA:
                DATA_hex_list.append(byte)
            DATA_add_hex_list = sum(DATA_hex_list)

            #
            try:
                SUM = (
                    sum(
                        [
                            HEAD_add_hex_list,
                            int.from_bytes(ADDR, byteorder="big"),
                            int.from_bytes(CMD, byteorder="big"),
                            int.from_bytes(LEN, byteorder="big"),
                            DATA_add_hex_list,
                        ]
                    )
                    % 256
                )  # 校验和
            except OverflowError:
                raise Exception("int too big to convert")
                # return False
            packet = HEAD + ADDR + CMD + LEN + DATA + bytes([SUM])  # 数据包
            self.port.write(packet)  # 将命令代码写入串口
            # return True  # 如果成功，则返回True，否则引发异常

        def release(self):
            self.send_data()

    # keyboard = ch9329Comm.keyboard.DataComm()
    keyboard = Keyboard(port=ser)

    # pass int to override.

    class Mouse(ch9329Comm.mouse.DataComm):
        @beartype
        def __init__(
            self, port: serial.Serial, screen_width: pos_int, screen_height: pos_int
        ):
            self.port = port
            super().__init__(screen_width=screen_width, screen_height=screen_height)

        @beartype
        def assert_inbound(self, x: non_neg_int, y: non_neg_int):
            assert x <= self.X_MAX, f"exceeding x limit ({self.X_MAX}): {x}"
            assert y <= self.Y_MAX, f"exceeding y limit ({self.Y_MAX}): {y}"

        @beartype
        def get_ctrl(self, x: int, y: int, button_codes: List[MouseButton], inbound: bool = True) -> int:
            if inbound:
                self.assert_inbound(x, y)
            ctrl: int = reduce_flags_to_bytes(button_codes, byte_length=1)
            return ctrl

        @beartype
        def send_data_absolute(self, x: non_neg_int, y: non_neg_int, button_codes: List[MouseButton] = []):
            ctrl = self.get_ctrl(x, y, button_codes)
            super().send_data_absolute(x, y, ctrl=ctrl, port=self.port)

        @beartype
        def send_data_relatively(self,  x: int, y: int, button_codes: List[MouseButton] = []):
            ctrl = self.get_ctrl(x, y, button_codes, inbound=False)
            super().send_data_relatively(x, y, ctrl=ctrl, port=self.port)

        @beartype
        def move_to_basic(self, x: non_neg_int, y: non_neg_int, button_codes: List[MouseButton] = []):

            ctrl = self.get_ctrl(x, y, button_codes)
            super().move_to_basic(x, y, ctrl=ctrl, port=self.port)

        @beartype
        def move_to(self, dest_x: non_neg_int, dest_y: non_neg_int, button_codes: List[MouseButton] = []):
            ctrl = self.get_ctrl(dest_x, dest_y, button_codes)
            super().move_to(dest_x, dest_y, ctrl=ctrl, port=self.port)

        @beartype
        # this is right click. we need to override this.
        def click(self, button):
            self.send_data_relatively(0, 0, button)
            time.sleep(random.uniform(0.1, 0.45))  # 100到450毫秒延迟
            self.send_data_relatively(0, 0)

    # mouse = ch9329Comm.mouse.DataComm(screen_width=1920, screen_height=1080)

    # (deprecated) monkey patch some method.

    # from types import MethodType

    # # to override instance methods.
    # keyboard.send_data = MethodType(send_data, keyboard)
    # keyboard.release = MethodType(release, keyboard)

else:
    raise Exception("Unknown device type: {deviceType}".format(
        deviceType=deviceType))

ser.close()
