cmd = "env XAUTHORITY=/tmp/.Xauthority ffmpeg -f x11grab -i :10 -f image2pipe pipe:1"

import subprocess

p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

while True:
    print(len(p.stdout.readline()))