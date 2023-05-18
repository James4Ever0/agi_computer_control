import subprocess

# run this using xvfb.

script_1 = 'xvfb-run --server-num=10 --listen-tcp --server-args="-screen 0 1280x800x24" -f /tmp/.Xauthority vboxsdl --startvm "Ubuntu 16.04"'

script_2 = 'env XAUTHORITY=/tmp/.Xauthorit ffplay -f x11grab -i :10'

import time

while True:
    p1 = subprocess.Popen(script_1)
    p2 = subproc