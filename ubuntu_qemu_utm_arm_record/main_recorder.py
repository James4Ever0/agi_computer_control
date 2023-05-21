import time
import subprocess
import datetime
import os
from utils import (
    set_redis_on,
    set_redis_off,
    set_prefix,
    check_redis_on,
    check_redis_off,
    PYTHON_EXECUTABLE,
    set_redis_off_on_exception,
    filepaths,
    MAX_RECORDING_COUNT,
)

set_prefix()

set_redis_off_on_exception()

MINIBREAK_SECONDS = 1
RECORD_SECONDS = 10

WAIT_TIMEOUT = 1

# how to signal multiple threads at once? use redis.
set_redis_off()
time.sleep(MINIBREAK_SECONDS)

RECORDERS = {"Audio": False, "Video": True, "HID": True}
# RECORDERS = {"Audio": False, "Video": False, "HID": True}
# RECORDERS = {"Audio": True, "Video": False, "HID": False}
# RECORDERS = {"Audio": True, "Video": True, "HID": True}
RANDOM_ACTOR = True

# keep last 30 recordings.
# will remove anything more than that.
rec_folders = [
    "{}{}".format(filepaths.target_prefix, p)
    for p in os.listdir(filepaths.target_prefix)
    if os.path.isdir("{}{}".format(filepaths.target_prefix, p))
]
rec_folders.sort(key=lambda p: -os.path.getmtime(p))
expired_rec_folders = rec_folders[MAX_RECORDING_COUNT:]
print("EXPIRED RECORDING FOLDER COUNT:", len(expired_rec_folders))
for p in expired_rec_folders:
    print("REMOVING EXPIRED RECORDING FOLDER:", p)
    os.system("rm -rf {}".format(p))

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
                [PYTHON_EXECUTABLE, "mouse_keyboard_record.py"]
            )
        if RECORDERS["Video"]:
            VideoRecorderProcess = subprocess.Popen(
                [PYTHON_EXECUTABLE, "video_record.py"]
            )
        if RECORDERS["Audio"]:
            AudioRecorderProcess = subprocess.Popen(
                [PYTHON_EXECUTABLE, "audio_record.py"]
            )

        if RANDOM_ACTOR:
            RandomActorProcess = subprocess.Popen(
                [PYTHON_EXECUTABLE, "random_actor_redis.py"]
            )
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
                video_exit_code = VideoRecorderProcess.wait(timeout=WAIT_TIMEOUT)
                exit_codes.append(video_exit_code)
            if RECORDERS["Audio"]:
                audio_exit_code = AudioRecorderProcess.wait(timeout=WAIT_TIMEOUT)
                exit_codes.append(audio_exit_code)
            if RANDOM_ACTOR:
                random_actor_exit_code = RandomActorProcess.wait(timeout=WAIT_TIMEOUT)
                exit_codes.append(random_actor_exit_code)
            print()
            print("EXIT CODES:")
            if RECORDERS["Audio"]:
                print("AUDIO - {}".format(audio_exit_code))
            if RECORDERS["Video"]:
                print("VIDEO - {}".format(video_exit_code))
            if RECORDERS["HID"]:
                print("HID - {}".format(hid_exit_code))
            if RANDOM_ACTOR:
                print("RANDOM_ACTOR - {}".format(random_actor_exit_code))
            print()
            if any([code != 0 for code in exit_codes]):
                # you may remove all temp files under recorder folder.
                for fpath in [
                    filepaths.hid_record,
                    filepaths.audio_record,
                    filepaths.video_record,
                    filepaths.video_record_script,
                    filepaths.video_timestamps,
                    filepaths.hid_timestamps,
                    filepaths.audio_timestamps,
                ]:
                    try:
                        os.remove(fpath)
                    except:
                        pass
                raise Exception("COMPUTER RECORDER HAS ABNORMAL EXIT CODE.")
            else:
                print("COMPUTER RECORDER EXIT NORMALLY")
                # required for ntfs.
                current_timestamp = (
                    datetime.datetime.now().isoformat().replace(":", "_")
                )
                records_folder = "{}{}".format(filepaths.target_prefix, current_timestamp)
                # records_folder = "{}{}".format(filepaths.prefix, current_timestamp)
                print("MOVING RECORDS TO: {}".format(records_folder))
                os.mkdir(records_folder)
                for fpath in [
                    filepaths.hid_record,
                    filepaths.audio_record,
                    filepaths.video_record,
                    filepaths.video_record_script,
                    filepaths.video_timestamps,
                    filepaths.hid_timestamps,
                    filepaths.audio_timestamps,
                ]:
                    os.system("mv {} {}".format(fpath, records_folder))
                # print("MAKING FINISHED INDICATOR")
                # os.system("touch {}".format(os.path.join(records_folder, "finished")))
                # or this is not needed. because it is always hard to savage things from a running instance.
        else:
            print("FAILED TO SET LOCK AS OFF.")
            print("FAILED AT FINAL CHECK.")
    else:
        print("FAILED TO SET LOCK AS ON.")
        print("FAILED AT INIT CHECK 2")
else:
    print("FAILED TO SET LOCK AS OFF.")
    print("FAILED AT INIT CHECK 1")
