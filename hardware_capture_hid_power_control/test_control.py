import serial
import beartype
from beartype.vale import Is
from typing import Annotated

# confusing!
serialDevices = {
    "power": "/dev/serial/by-id/usb-1a86_5523-if00-port0",
    "hid": "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
}

# deviceType = "power"
deviceType = "hid"  # 为了保证数据能正常传输，两条数据发送间隔最低要有5ms 的延时；意思就是你发送一个数据后延时5ms 再发下一条数据。

ser = serial.Serial(serialDevices[deviceType], timeout=0.01,
                    **({'baudrate': 57600} if deviceType == 'hid' else {}))
print("Serial device: %s" % deviceType)

# print(dir(ser))
# ['BAUDRATES', 'BAUDRATE_CONSTANTS', 'BYTESIZES', 'PARITIES', 'STOPBITS', '_SAVED_SETTINGS', ..., '_baudrate', '_break_state', '_bytesize', '_checkClosed', '_checkReadable', '_checkSeekable', '_checkWritable', '_dsrdtr', '_dtr_state', '_exclusive', '_inter_byte_timeout', '_parity', '_port', '_reconfigure_port', '_reset_input_buffer', '_rs485_mode', '_rts_state', '_rtscts', '_set_rs485_mode', '_set_special_baudrate', '_stopbits', '_timeout', '_update_break_state', '_update_dtr_state', '_update_rts_state', '_write_timeout', '_xonxoff', 'applySettingsDict', 'apply_settings', 'baudrate', 'break_condition', 'bytesize', 'cancel_read', 'cancel_write', 'cd', 'close', 'closed', 'cts', 'dsr', 'dsrdtr', 'dtr', 'exclusive', 'fd', 'fileno', 'flush', 'flushInput', 'flushOutput', 'getCD', 'getCTS', 'getDSR', 'getRI', 'getSettingsDict', 'get_settings', 'inWaiting', 'in_waiting', 'interCharTimeout', 'inter_byte_timeout', 'iread_until', 'isOpen', 'is_open', 'isatty', 'name', 'nonblocking', 'open', 'out_waiting', 'parity', 'pipe_abort_read_r', 'pipe_abort_read_w', 'pipe_abort_write_r', 'pipe_abort_write_w', 'port', 'portstr', 'read', 'read_all', 'read_until', 'readable', 'readall', 'readinto', 'readline', 'readlines', 'reset_input_buffer', 'reset_output_buffer', 'ri', 'rs485_mode', 'rts', 'rtscts', 'seek', 'seekable', 'sendBreak', 'send_break', 'setDTR', 'setPort', 'setRTS', 'set_input_flow_control', 'set_low_latency_mode', 'set_output_flow_control', 'stopbits', 'tell', 'timeout', 'truncate', 'writable', 'write', 'writeTimeout', 'write_timeout', 'writelines', 'xonxoff']

# import rich
# rich.print(ser.__dict__)]

# print(ser.name) # /dev/serial/by-id/usb-1a86_5523-if00-port0

# ser.write(b"hello")

@bea
def write_and_read(_bytes: bytes):
    ser.write(_bytes)
    print(f"w> {repr(_bytes)}")
    res = ser.readall()
    print(f"r> {repr(res)}")


if deviceType == "power":
    # will reset on reboot
    channel = 1  # CH3 does not exist. CH2 is placeholder. (virtually working)
    # channel = 2
    state = "ON"
    # state = "OFF"

    write_and_read(f"CH{channel}=?".encode())
    write_and_read(f"CH{channel}={state}".encode())
    write_and_read(f"CH{channel}=?".encode())
elif deviceType == "hid":
    commonHeader = b"\x57\xab"
    modifyIDHeader = commonHeader+b"\x10"  # +4bits, (2bits VID, 2bits PID)
    keyboardHeader = commonHeader+b"\x01"  # +8bits
    mouseRelativeHeader = commonHeader+b"\x02"  # +4bits
    mouseMultimediaHeader = commonHeader+b"\x03"  # +(2 or 4)bits
    mouseAbsoluteHeader = commonHeader+b"\x04"  # +4bits
    def changeID(vid:bytes, pid:bytes):
        assert 
else:
    raise Exception("Unknown device type: {deviceType}".format(
        deviceType=deviceType))

ser.close()
