# import subprocess

# # run this using xvfb.
import os

# script_1 = 'xvfb-run --server-num=10 --listen-tcp --server-args="-screen 0 1280x800x24" -f /tmp/.Xauthority vboxsdl --startvm "Ubuntu 16.04"'

# script_2 = 'env XAUTHORITY=/tmp/.Xauthority ffplay -f x11grab -i :10'

# script_3 = 'vboxmanage controlvm "Ubuntu 16.04" poweroff'

script_4 = "bash x11grab_loop.sh"
import time
# import progressbar

while True:
    time.sleep(1)
    os.system("cd .") # FIX: input/output error
    os.system(script_4)

# while True:
#     p1 = subprocess.run(script_1)
#     p2 = subprocess.run(script_2)

#     for _ in progressbar.progressbar(range(100)):
#         time.sleep(1)
    
#     os.system(script_3)
    