# make sure we have the screen size.
import os
from utils import filepaths, PYTHON_EXECUTABLE, set_redis_off_on_exception
import mss

set_redis_off_on_exception()

s = mss.mss()
img = s.grab(s.monitors[0])
size = img.size
print("IMAGE SIZE?", size)
video_size = "{}x{}".format(size.width, size.height)


pix_fmt = "bgr0"
framerate = 30
codec = "libx264"

script_content = "{} pyscreenshot_output.py | ffmpeg -y -f rawvideo -pix_fmt {} -s {} -i - -r {} -c:v {} {}".format(
    PYTHON_EXECUTABLE, pix_fmt, video_size, framerate, codec, filepaths.video_record
)

with open(filepaths.video_record_script, "w+") as f:
    f.write(script_content)


os.system("bash {}".format(filepaths.video_record_script))
