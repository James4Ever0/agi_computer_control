import time
import subprocess
from utils import set_redis_on, set_redis_off, check_redis_on, check_redis_off, PYTHON_EXECUTABLE, set_redis_off_on_exception

set_redis_off_on_exception()

MINIBREAK_SECONDS = 1
RECORD_SECONDS = 10

WAIT_TIMEOUT = 1

# how to signal multiple threads at once? use redis.
set_redis_off()
time.sleep(MINIBREAK_SECONDS)

# RECORDERS = {"Audio": False, "Video": True, "HID": True}
# RECORDERS = {"Audio": False, "Video": False, "HID": True}
RECORDERS = {"Audio": True, "Video": False, "HID": False}
# RECORDERS = {"Audio": True, "Video": True, "HID": True}

if all([signal is not True for _, signal in RECORDERS.items()]):
    raise Exception("Should at least use one recorder.")
else:
    for key, value in RECORDERS.items():
        if value:
            print("Recording: %s" % key)
if check_redis_off():
    set_redis_on()
    time.sleep(MINIBREAK_SECONDS)
    if check_redis_on():
        print("EXECUTING MAIN PROCESSES")
        print("RECORD LENGTH: {} secs".format(RECORD_SECONDS))
        # execute subcommands. (subprocess)
        if RECORDERS["HID"]:
            HIDRecorderProcess = subprocess.Popen(
                [PYTHON_EXECUTABLE, 'mouse_keyboard_record.py'])
        if RECORDERS["Video"]:
            VideoRecorderProcess = subprocess.Popen(
                [PYTHON_EXECUTABLE, 'video_record.py'])
        if RECORDERS["Audio"]:
            AudioRecorderProcess = subprocess.Popen(
                [PYTHON_EXECUTABLE, 'audio_record.py'])
        # time.sleep(RECORD_SECONDS)
        for _ in range(RECORD_SECONDS):
            time.sleep(1)
            if check_redis_off():
                print("Abnormal recorder exit detected.")
                print("Abort main recorder.")
                break
        print("EXITING.")
        print("SET LOCK AS OFF.")
        set_redis_off()
        time.sleep(MINIBREAK_SECONDS)
        if check_redis_off():
            exit_codes = []
            if RECORDERS["HID"]:
                hid_exit_code = HIDRecorderProcess.wait(timeout=WAIT_TIMEOUT)
                exit_codes.append(hid_exit_code)
            if RECORDERS["Video"]:
                video_exit_code = VideoRecorderProcess.wait(
                    timeout=WAIT_TIMEOUT)
                exit_codes.append(video_exit_code)
            if RECORDERS["Audio"]:
                audio_exit_code = AudioRecorderProcess.wait(
                    timeout=WAIT_TIMEOUT)
                exit_codes.append(audio_exit_code)
            print()
            print("EXIT CODES:")
            if RECORDERS["Audio"]:
                print(f"AUDIO - {audio_exit_code}")
            if RECORDERS["Video"]:
                print(f'VIDEO - {video_exit_code}')
            if RECORDERS["HID"]:
                print(f'HID - {hid_exit_code}')
            print()
            if any([code != 0 for code in exit_codes]):
                raise Exception("COMPUTER RECORDER HAS ABNORMAL EXIT CODE.")
            else:
                print("COMPUTER RECORDER EXIT NORMALLY")
        else:
            print("FAILED TO SET LOCK AS OFF.")
            print("FAILED AT FINAL CHECK.")
    else:
        print("FAILED TO SET LOCK AS ON.")
        print("FAILED AT INIT CHECK 2")
else:
    print("FAILED TO SET LOCK AS OFF.")
    print("FAILED AT INIT CHECK 1")
