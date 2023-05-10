import time
import subprocess
from utils import set_redis_on, set_redis_off, check_redis_on, check_redis_off

MINIBREAK_SECONDS = 1
RECORD_SECONDS = 10

# how to signal multiple threads at once? use redis.
set_redis_off()
time.sleep(MINIBREAK_SECONDS)

if check_redis_off():
    set_redis_on()
    time.sleep(MINIBREAK_SECONDS)
    if check_redis_on():
        print("EXECUTING MAIN PROCESSES")
        print("RECORD LENGTH: {} secs".format(RECORD_SECONDS))
        # execute subcommands. (subprocess)
        subprocess.Popen(['python3','mouse_keyboard_record.py'])
        time.sleep(RECORD_SECONDS)
        print("EXITING.")
        print("SET LOCK AS OFF.")
        set_redis_off()
        time.sleep(MINIBREAK_SECONDS)
        if check_redis_off():
            print("HID RECORDER EXIT NORMALLY")
        else:
            print("FAILED TO SET LOCK AS OFF.")
            print("FAILED AT FINAL CHECK.")
    else:
        print("FAILED TO SET LOCK AS ON.")
        print("FAILED AT INIT CHECK 2")
else:
    print("FAILED TO SET LOCK AS OFF.")
    print("FAILED AT INIT CHECK 1")
