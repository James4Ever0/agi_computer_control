#import pyautogui
#
#img = pyautogui.screenshot()
##print(img)
##print(dir(img))
#img.save("output.png")
#
#img_bytes = img.tobytes()
#
#print(len(img_bytes), type(img_bytes))
#
#import sys
#
## maybe it is just completely zero.
#
#with open(sys.stdout.fileno(), 'wb', closefd=False) as stdout:
#    #stdout.write(b"hello world")
#    stdout.write(img_bytes)
#    stdout.flush()

# this sucks. this is wayland. hell. what will happen on pynput?
# i'd better stick to x11vnc instead. but is this x11?
# import pyscreenshot

import sys

import mss

s = mss.mss()

with open(sys.stdout.fileno(), 'wb', closefd=False) as stdout:
    for _ in range(100):
        # img = pyscreenshot.grab()
        # img_bytes = img.tobytes()
        img = s.grab(s.monitors[0])
        img_bytes = img.raw
        stdout.write(img_bytes)
        stdout.flush()
