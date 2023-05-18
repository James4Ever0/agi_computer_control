
from xvfbwrapper import Xvfb

vdisplay = Xvfb(width=1280, height=740)
vdisplay.start()

import os

try:
    # launch stuff inside virtual display here.
    os.system('bash start_vm.sh')
    import time
    import progressbar
    for _ in progressbar.progressbar(range(1000)):
        time.sleep(1)
finally:
    vdisplay.stop()