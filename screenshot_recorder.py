# shall you make the screen recording and the keylogging separate.
# or you just want to screw things up so much.

import mss
screenshot_factory = mss.mss()


import os
import shutil

imagedir = "screenshots"
if os.path.exists(imagedir):
    shutil.rmtree(imagedir)
os.mkdir(imagedir)
import time
timestep= 0.03

world_start = time.time()
import jsonlines
from config import screenshotLogPath
with jsonlines.open(screenshotLogPath, 'w') as w:
    for loopIndex in range(500):
        time.sleep(timestep)
        image = screenshot_factory.grab(screenshot_factory.monitors[0])
        imagebytes = image.raw
        imagePath = f"{imagedir}/{loopIndex}.raw" # no compression needed.
        with open(imagePath, 'wb') as f:
            f.write(imagebytes)
        print("WRITING TO:", imagePath)
        current_time = time.time()
        print("TIME?", current_time-world_start)
        w.write({"timeStamp": current_time, "imagePath": imagePath, "imageSize": image.size})