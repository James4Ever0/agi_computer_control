import os
import time

pid = os.getpid()

# killed by os.kill!
while True:
    print("process pid:", pid)
    time.sleep(1)
