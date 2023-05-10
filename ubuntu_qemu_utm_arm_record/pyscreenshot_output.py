# import pyautogui
#
# img = pyautogui.screenshot()
##print(img)
##print(dir(img))
# img.save("output.png")
#
# img_bytes = img.tobytes()
#
# print(len(img_bytes), type(img_bytes))
#
# import sys
#
## maybe it is just completely zero.
#
# with open(sys.stdout.fileno(), 'wb', closefd=False) as stdout:
#    #stdout.write(b"hello world")
#    stdout.write(img_bytes)
#    stdout.flush()

# this sucks. this is wayland. hell. what will happen on pynput?
# i'd better stick to x11vnc instead. but is this x11?
# import pyscreenshot

import mss
import sys
from utils import TimestampedContext, filepaths, check_redis_on, check_redis_off

s = mss.mss()

if check_redis_on():
    with TimestampedContext(filepaths.video_timestamps) as t:
        with open(sys.stdout.fileno(), "wb", closefd=False) as stdout:
            while check_redis_off() is True:
                # img = pyscreenshot.grab()
                # img_bytes = img.tobytes()
                img = s.grab(s.monitors[0])
                img_bytes = img.raw
                stdout.write(img_bytes)
                stdout.flush()
                t.commit()
else:
    raise Exception("VideoRecorder: Can't start. Redis signal is off.".upper())
