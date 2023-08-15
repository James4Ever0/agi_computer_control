from typing import Callable
import math
import serial
from beartype import beartype
from beartype.vale import Is
from typing import Annotated
from typing_extensions import TypeAlias
import sys
sys.path.append("../")
from hid_utils import *

# use xephyr (leafpad, fullscreen) for unit test.

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from strenum import StrEnum

# for branching; ref: https://beartype.readthedocs.io/en/latest/api_door/
from beartype.door import is_bearable
from enum import Enum, auto, Flag
import time
import random
from functools import reduce
from typing import Union, List, Literal, Tuple
from common_keycodes import KeyLiteralToKCOMKeycode, HIDActionTypes
import inspect



# confusing!

# TODO: unit test underway

@beartype
def get_scroll_code(c_scroll: movement) -> one_byte:
    if c_scroll < 0:
        c_scroll = -c_scroll + 0x80
    return c_scroll.to_bytes()


class DeviceType(StrEnum):
    power = auto()
    kcom2 = auto()
    kcom3 = auto()
    ch9329 = auto()


serialDevices = {
    DeviceType.power: "/dev/serial/by-id/usb-1a86_5523-if00-port0",
    # kcom2/kcom3 & ch9329 not distinguishable by id (all ch340).
    DeviceType.kcom2: (ch340 := "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"),
    # another hid device will be: ch9329
    DeviceType.kcom3: ch340,
    DeviceType.ch9329: ch340,
}

deviceType = DeviceType.power
# deviceType = DeviceType.ch9329
# deviceType = DeviceType.kcom3
# deviceType = DeviceType.kcom2  # 为了保证数据能正常传输，两条数据发送间隔最低要有5ms 的延时；意思就是你发送一个数据后延时5ms 再发下一条数据。

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



@beartype
def reduce_flags_to_bytes(  # force this to be non-empty!
    flags: List[Flag],
    # flags: Annotated[List[Flag], Is[lambda l: len(l) > 0]],
    byteorder: Literal["little", "big"] = "little",
    byte_length: Union[int, Ellipsis] = ...,
):
    # def reduce_flags_to_bytes(opcodes: List[Union[one_byte, two_bytes]]):
    if flags == []:
        assert is_bearable(
            pos_int, byte_length), f"invalid byte_length (positive integer): {byte_length}"
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

elif deviceType in [DeviceType.kcom2, DeviceType.kcom3]:
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
        control_codes: List[ControlCode] = [ControlCode.NULL],
        key_literals: Annotated[
            List[HIDActionTypes.keys], Is[lambda l: len(
                l) <= 6 and len(l) >= 0]
        ] = [],
    ):  # check for "HID Usage ID"
        reserved_byte = b"\x00"
        # control_code = reduce_flags_to_bytes(control_codes)
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
    def get_rel_code_kcom(c_rel: movement):
        if c_rel < 0:
            c_rel = 0xFF + c_rel
        return c_rel.to_bytes()

    @beartype
    def mouse_common(
        x_code: Union[two_bytes, one_byte],
        y_code: Union[two_bytes, one_byte],
        scroll: movement,
        kcom_flag: Literal[
            KCOMHeader.mouseRelativeHeader, KCOMHeader.mouseAbsoluteHeader
        ],
        button_codes: List[MouseButton] = [MouseButton.NULL],
    ):
        scroll_code = get_scroll_code(scroll)
        # button_code = reduce_flags_to_bytes(button_codes)
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
        x: movement, y: movement, scroll: movement, button_codes: List[MouseButton] = [MouseButton.NULL]
    ):
        x_code = get_rel_code_kcom(x)
        y_code = get_rel_code_kcom(y)

        mouse_common(
            x_code,
            y_code,
            scroll,
            kcom_flag=KCOMHeader.mouseRelativeHeader,
            button_codes=button_codes,
        )

    def get_abs_code(c_abs, res):
        return int((4096 * c_abs) / res).to_bytes(2, byteorder="little")

    @beartype
    def mouse_absolute(
        coordinate: Tuple[non_neg_int, non_neg_int],
        resolution: Tuple[pos_int, pos_int],
        scroll: movement,
        button_codes: List[MouseButton] = [MouseButton.NULL],
    ):
        """
        coordinate: (x_abs, y_abs)
        resolution: (width, height)
        """

        (x_abs, y_abs) = coordinate
        (width, height) = resolution

        assert x_abs <= width, f"Invalid x: {x_abs}\nWidth: {width}"
        assert y_abs <= height, f"Invalid y: {y_abs}\nHeight: {height}"

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

    # @beartype
    # def multimedia_raw(data_code: Union[two_bytes, four_bytes]):

    @beartype
    def multimedia(keys: Union[List[ACPIKey], List[MultimediaKey]] = []):
        if len(keys) == 0:  # clear all multimedia keys.
            multimedia(keys=[ACPIKey.Null])
            multimedia(keys=[MultimediaKey.Null])
            return
        isMultimediaKeys = is_bearable(keys, List[MultimediaKey])
        byte_length = 3 if isMultimediaKeys else 1
        # key_code = reduce_flags_to_bytes(keys)
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
    # import parse
    from types import MethodType
    # from types import MethodWrapperType

    @beartype
    class CH9329Util:
        def __init__(self, port: serial.Serial, **kwargs):
            self.port = port
            super_class_init = getattr(super(), '__init__', None)
            if super_class_init:
                # not method-wrapper.
                if isinstance(super_class_init, MethodType):
                    # sclass_init_str = str(super_class_init)
                    # sclass_str = str(super())
                    # sclass_parsed = parse.parse("<super: <class '{self}'>, <{base} object>>", sclass_str)
                    # base_init_str =
                    # self.super_class = super_class_init(**kwargs)
                    super().__init__(**kwargs)

        def communicate(self, DATA: Annotated[bytes, Is[lambda b: len(b) > 0]], CMD: one_byte,  LEN: one_byte):
            # 将字符转写为数据包
            HEAD = b"\x57\xAB"  # 帧头
            ADDR = b"\x00"  # 地址
            # CMD = b"\x02"  # 命令
            # LEN = b"\x08"  # 数据长度
            data_length = ord(LEN)

            # 控制键
            # control_byte = reduce_flags_to_bytes(control_codes, byte_length=1)
            # DATA += control_byte
            # # if ctrl == '':
            # #     DATA += b'\x00'
            # # elif isinstance(ctrl, int):
            # #     DATA += bytes([ctrl])
            # # else:
            # #     DATA += self.control_button_hex_dict[ctrl]

            # # DATA固定码
            # DATA += b"\x00"

            # 读入data
            # for i in range(0, len(data), 2):
            #     DATA += self.normal_button_hex_dict[data[i:i + 2]]
            # for key_literal in key_literals:
            #     DATA += KeyLiteralToKCOMKeycode(key_literal)
            if len(DATA) < data_length:
                DATA += b"\x00" * (data_length - len(DATA))
            else:
                DATA = DATA[:data_length]

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

            SUM = self.checksum(HEAD_add_hex_list, ADDR,
                                CMD, LEN, DATA_add_hex_list)
            packet = HEAD + ADDR + CMD + LEN + DATA + bytes([SUM])  # 数据包
            self.port.write(packet)  # 将命令代码写入串口
            # return True  # 如果成功，则返回True，否则引发异常

        def checksum(
            self,
            HEAD_add_hex_list: int,
            ADDR: bytes,
            CMD: bytes,
            LEN: bytes,
            DATA_add_hex_list: int,
        ):
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
            except Exception as e:
                print("int too big to convert")
                raise e
                # return False
            return SUM

    @beartype
    class Multimedia(CH9329Util):
        def send_data(self, keys: Union[List[ACPIKey], List[MultimediaKey]] = []):
            if len(keys) == 0:  # clear all multimedia keys.
                Multimedia.send_data(keys=[ACPIKey.Null])
                Multimedia.send_data(keys=[MultimediaKey.Null])
                return
            isMultimediaKeys = is_bearable(keys, List[MultimediaKey])

            CMD = b"\x03"  # 命令
            LEN = b"\x04" if isMultimediaKeys else b"\x02"  # 数据长度
            byte_length = 3 if isMultimediaKeys else 1

            key_code = reduce_flags_to_bytes(keys, byte_length=byte_length)
            DATA = (b"\x02" if isMultimediaKeys else b"\x01") + key_code  # 数据

            self.communicate(DATA, CMD, LEN)

        def release(self):
            self.send_data()

    # ref: https://github.com/beijixiaohu/CH9329_COMM
    @beartype
    class Keyboard(CH9329Util):
        # def __init__(
        #     self,
        #     port: serial.Serial,
        # ):
        #     self.port = port

        def send_data(
            self,
            # [ControlCode.NULL] or [], both works
            control_codes: List[ControlCode] = [ControlCode.NULL],
            key_literals: Annotated[
                List[HIDActionTypes.keys], Is[lambda l: len(
                    l) <= 8 and len(l) >= 0]
            ] = [],
        ):
            # 将字符转写为数据包
            # HEAD = b"\x57\xAB"  # 帧头
            # ADDR = b"\x00"  # 地址
            CMD = b"\x02"  # 命令
            LEN = b"\x08"  # 数据长度
            DATA = b""  # 数据

            # 控制键
            # control_byte = reduce_flags_to_bytes(control_codes)
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
            self.communicate(DATA, CMD, LEN)
            # if len(DATA) < 8:
            #     DATA += b"\x00" * (8 - len(DATA))
            # else:
            #     DATA = DATA[:8]

            # # 分离HEAD中的值，并计算和
            # HEAD_hex_list = []
            # for byte in HEAD:
            #     HEAD_hex_list.append(byte)
            # HEAD_add_hex_list = sum(HEAD_hex_list)

            # # 分离DATA中的值，并计算和
            # DATA_hex_list = []
            # for byte in DATA:
            #     DATA_hex_list.append(byte)
            # DATA_add_hex_list = sum(DATA_hex_list)

            # #
            # try:
            #     SUM = (
            #         sum(
            #             [
            #                 HEAD_add_hex_list,
            #                 int.from_bytes(ADDR, byteorder="big"),
            #                 int.from_bytes(CMD, byteorder="big"),
            #                 int.from_bytes(LEN, byteorder="big"),
            #                 DATA_add_hex_list,
            #             ]
            #         )
            #         % 256
            #     )  # 校验和
            # except OverflowError:
            #     raise Exception("int too big to convert")
            #     # return False
            # packet = HEAD + ADDR + CMD + LEN + DATA + bytes([SUM])  # 数据包
            # self.port.write(packet)  # 将命令代码写入串口
            # # return True  # 如果成功，则返回True，否则引发异常

        def release(self):
            self.send_data()

    # keyboard = ch9329Comm.keyboard.DataComm()
    keyboard = Keyboard(port=ser)  # TODO: multimedia key support

    # pass int to override.
    @beartype
    class Mouse(CH9329Util, ch9329Comm.mouse.DataComm):
        def __init__(
            self, port: serial.Serial, screen_width: pos_int, screen_height: pos_int
        ):
            # self.port = port
            initargs = dict(screen_width=screen_width,
                            screen_height=screen_height)
            super().__init__(port=port, **initargs)
            self.super_instance = ch9329Comm.mouse.DataComm(**initargs)

        # TODO: scroll support

        def assert_inbound(self, x: non_neg_int, y: non_neg_int):
            assert x <= self.X_MAX, f"exceeding x limit ({self.X_MAX}): {x}"
            assert y <= self.Y_MAX, f"exceeding y limit ({self.Y_MAX}): {y}"

        def get_ctrl(
            self, x: int, y: int, button_codes: List[MouseButton], inbound: bool
        ) -> int:
            if inbound:
                self.assert_inbound(x, y)
            ctrl: int = reduce_flags_to_bytes(button_codes, byte_length=1)
            return ctrl

        def call_super_method(
            self,
            funcName: str,
            x: int,
            y: int,
            button_codes: List[MouseButton] = [MouseButton.NULL],
            inbound: bool = True,
            use_super_instance: bool = False,
        ):
            ctrl = self.get_ctrl(x, y, button_codes, inbound=inbound)
            ret = (
                self.super_instance
                if use_super_instance
                else getattr(super(), funcName)
            )(x, y, ctrl=ctrl, port=self.port)
            if ret == False:
                raise Exception(
                    "Error calling super method: {}".format(funcName))

        def send_data_absolute(
            self,
            x: non_neg_int,
            y: non_neg_int,
            scroll: movement,
            button_codes: List[MouseButton] = [MouseButton.NULL],
        ):
            ctrl = self.get_ctrl(x, y, button_codes=button_codes, inbound=True)
            # currentFuncName = inspect.currentframe().f_code.co_name
            # self.call_super_method(currentFuncName, x, y, button_codes)

            # 将字符转写为数据包
            # HEAD = b"\x57\xAB"  # 帧头
            # ADDR = b"\x00"  # 地址
            CMD = b"\x04"  # 命令
            LEN = b"\x07"  # 数据长度
            DATA = bytearray(b"\x02")  # 数据

            # 鼠标按键
            # if ctrl == "":
            #     DATA.append(0)
            # elif isinstance(ctrl, int):
            DATA.append(ctrl)
            # else:
            #     DATA += self.hex_dict[ctrl]

            # 坐标
            X_Cur = (4096 * x) // self.X_MAX
            Y_Cur = (4096 * y) // self.Y_MAX
            DATA += X_Cur.to_bytes(2, byteorder="little")
            DATA += Y_Cur.to_bytes(2, byteorder="little")

            DATA += get_scroll_code(scroll)
            self.communicate(bytes(DATA), CMD, LEN)

            # if len(DATA) < 7:
            #     DATA += b"\x00" * (7 - len(DATA))
            # else:
            #     DATA = DATA[:7]

            # # 分离HEAD中的值，并计算和
            # HEAD_hex_list = list(HEAD)
            # HEAD_add_hex_list = sum(HEAD_hex_list)

            # # 分离DATA中的值，并计算和
            # DATA_hex_list = list(DATA)
            # DATA_add_hex_list = sum(DATA_hex_list)

            # try:
            #     SUM = (
            #         sum(
            #             [
            #                 HEAD_add_hex_list,
            #                 int.from_bytes(ADDR, byteorder="big"),
            #                 int.from_bytes(CMD, byteorder="big"),
            #                 int.from_bytes(LEN, byteorder="big"),
            #                 DATA_add_hex_list,
            #             ]
            #         )
            #         % 256
            #     )  # 校验和
            # except OverflowError:
            #     raise Exception("int too big to convert")
            # packet = HEAD + ADDR + CMD + LEN + DATA + bytes([SUM])  # 数据包
            # self.port.write(packet)  # 将命令代码写入串口
            # # return True  # 如果成功，则返回True，否则引发异常

        def send_data_relatively(
            self, x: int, y: int, scroll: movement, button_codes: List[MouseButton] = [MouseButton.NULL]
        ):
            ctrl = self.get_ctrl(
                x, y, button_codes=button_codes, inbound=False)
            # currentFuncName = inspect.currentframe().f_code.co_name
            # self.call_super_method(currentFuncName, x, y,
            #                        button_codes, inbound=False)

            # 将字符转写为数据包
            # HEAD = b"\x57\xAB"  # 帧头
            # ADDR = b"\x00"  # 地址
            CMD = b"\x05"  # 命令
            LEN = b"\x05"  # 数据长度
            DATA = bytearray(b"\x01")  # 数据

            # 鼠标按键
            # if ctrl == "":
            #     DATA.append(0)
            # elif isinstance(ctrl, int):
            DATA.append(ctrl)
            # else:
            #     DATA += self.hex_dict[ctrl]

            # x坐标
            if x == 0:
                DATA.append(0)
            elif x < 0:
                DATA += (0 - abs(x)).to_bytes(1, byteorder="big", signed=True)
            else:
                DATA += x.to_bytes(1, byteorder="big", signed=True)

            # y坐标，这里为了符合坐标系直觉，将<0改为向下，>0改为向上
            # y = - y
            # change your ass.
            # after doing this, we shall perform unittests, to ensure its integrity.
            if y == 0:
                DATA.append(0)
            elif y < 0:
                DATA += (0 - abs(y)).to_bytes(1, byteorder="big", signed=True)
            else:
                DATA += y.to_bytes(1, byteorder="big", signed=True)

            DATA += get_scroll_code(scroll)

            DATA += b"\x00" * (5 - len(DATA)) if len(DATA) < 5 else DATA[:5]
            self.communicate(bytes(DATA), CMD, LEN)

            # # 分离HEAD中的值，并计算和
            # HEAD_hex_list = list(HEAD)
            # HEAD_add_hex_list = sum(HEAD_hex_list)

            # # 分离DATA中的值，并计算和
            # DATA_hex_list = list(DATA)
            # DATA_add_hex_list = sum(DATA_hex_list)

            # try:
            #     SUM = (
            #         sum(
            #             [
            #                 HEAD_add_hex_list,
            #                 int.from_bytes(ADDR, byteorder="big"),
            #                 int.from_bytes(CMD, byteorder="big"),
            #                 int.from_bytes(LEN, byteorder="big"),
            #                 DATA_add_hex_list,
            #             ]
            #         )
            #         % 256
            #     )  # 校验和
            # except OverflowError:
            #     raise Exception("int too big to convert")
            # packet = HEAD + ADDR + CMD + LEN + DATA + bytes([SUM])  # 数据包
            # self.port.write(packet)  # 将命令代码写入串口
            # # return True  # 如果成功，则返回True，否则引发异常

        def move_to_basic(
            self, x: non_neg_int, y: non_neg_int, button_codes: List[MouseButton] = [MouseButton.NULL]
        ):
            currentFuncName = inspect.currentframe().f_code.co_name
            self.call_super_method(
                currentFuncName, x, y, button_codes, use_super_instance=True
            )

        def move_to(
            self,
            dest_x: non_neg_int,
            dest_y: non_neg_int,
            button_codes: List[MouseButton] = [MouseButton.NULL],
        ):
            currentFuncName = inspect.currentframe().f_code.co_name
            self.call_super_method(
                currentFuncName, dest_x, dest_y, button_codes, use_super_instance=True
            )

        # this is right click. we need to override this.
        def click(self, button: MouseButton, get_delay: Callable[[], float] = lambda: random.uniform(0.1, 0.45)):
            self.send_data_relatively(0, 0, [button])
            time.sleep(get_delay())  # 100到450毫秒延迟
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
