# timestamp, redis, constants.


# it is the main recorder which will pack all recordings into hdf5 file format after success.

class filepaths:
    hid_record = "hid_record.jsonl"
    audio_record = "audio_record.wav"
    video_record = "video_record.mp4"
    
    video_timestamps = "video_timestamps.json"
    hid_timestamps = "hid_timestamps.json"
    audio_timestamps = "audio_timestamps.json"


lock_key = "HID_MAIN_RECORDER_LOCK"
sig_on = "ON"
sig_off = "OFF"

timestep = 0.03
# filePath = "states.jsonl"


import os
import redis
import functools


@functools.lru_cache(maxsize=1)
def get_redis_client():
    r = redis.Redis(host="localhost", port=6379, db=0)
    return r


def set_redis_off():
    r = get_redis_client()
    r.set(lock_key, sig_off)


def set_redis_on():
    r = get_redis_client()
    r.set(lock_key, sig_on)


from typing import Union


def get_redis_value() -> Union[None, str]:
    r = get_redis_client()
    val = r.get(lock_key)
    if val:
        val = val.decode("utf-8")
    return val


def check_redis_off():
    return get_redis_value() == sig_off


def check_redis_on():
    return get_redis_value() == sig_on


import time
import json


class TimestampedLogCreater:
    def __init__(self, file_name, indent_output=True):
        self.file_name = file_name
        self.timestamp_list = []
        self.indent_output = indent_output

    def clear(self):
        self.timestamp_list = []

    def commit(self):
        timestamp = time.time()
        self.timestamp_list.append(timestamp)

    def read(self):
        with open(self.file_name, "r") as f:
            string = f.read()
            data = json.loads(string)
        for elem in data:
            assert type(elem) == float
        self.timestamp_list = data
        return data

    def write(self):
        with open(self.file_name, "w+") as f:
            string = json.dumps(
                self.timestamp_list, indent=4 if self.indent_output else None
            )
            f.write(string)
        print("TIMESTAMPED LOG WRITE TO: {}".format(self.file_name))


# create some context manager, for exporting the timestamped log.
class TimestampedContext:
    def __init__(self, file_name):
        self.file_name = file_name
        print("INIT TIMESTAMPED CONTEXT AT: {}".format(self.file_name))
        if os.path.exists(file_name):
            if os.path.isfile(file_name):
                print("REMOVING OLD TIMESTAMPED CONTEXT AT: {}".format(self.file_name))
                os.remove(file_name)
            else:
                raise Exception(
                    "PATH {} EXISTS AND IS NOT A FILE.".format(self.file_name)
                )
        self.mTimestampedLogCreater = TimestampedLogCreater(self.file_name)

    def __enter__(self):
        print("ENTER TIMESTAMPED CONTEXT AT: {}".format(self.file_name))
        return self.mTimestampedLogCreater

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # we don't have to take care of this.
        if exc_type == None:
            self.mTimestampedLogCreater.write()
        else:
            print("ERROR IN TIMESTAMPED CONTEXT")
            print("NOT WRITING TIMESTAMPED LOG AT {}".format(self.file_name))
        print("EXITING TIMESTAMPED CONTEXT")
