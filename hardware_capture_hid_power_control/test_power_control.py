import serial

ser = serial.Serial("/dev/serial/by-id/usb-1a86_5523-if00-port0")
# print(dir(ser))
import rich
# rich.print(ser.__dict__)]

ser.