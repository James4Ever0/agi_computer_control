
from xvfbwrapper import Xvfb

# it sucks!

# vdisplay = Xvfb(display=3)
# vdisplay.start()

import os

cmd = 'vboxsdl --startvm "Ubuntu 16.04"' 

with Xvfb(display=3) as vdisplay: # not working...
    os.system(cmd)
# try:
#     # launch stuff inside virtual display here.
#     # os.system('bash start_vm.sh')
#     import time
#     import progressbar
#     for _ in progressbar.progressbar(range(1000)):
#         time.sleep(1)
# finally:
#     vdisplay.stop()