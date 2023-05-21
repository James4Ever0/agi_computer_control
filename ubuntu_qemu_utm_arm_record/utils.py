# timestamp, redis, constants.
# PYTHON_EXECUTABLE = 'python3' # just in case.

import json
import time
from typing import Union
import functools
import redis
import os
import sys

PYTHON_EXECUTABLE = sys.executable


@functools.lru_cache(maxsize=1)
def get_redis_client():
    r = redis.Redis(host="localhost", port=6379, db=0)
    return r

# it is the main recorder which will pack all recordings into hdf5 file format after success.

PREFIX_KEY = "RECORD_PREFIX"
import uuid

def set_prefix():
    r = get_redis_client()
    prefix = "/tmp/{}/".format(str(uuid.uuid4()).replace("-", "_"))
    print("SET PREFIX: {}".format(prefix))
    os.mkdir(prefix)
    r.set(PREFIX_KEY, prefix)

def get_prefix():
    r = get_redis_client()
    val = r.get(PREFIX_KEY)
    if val:
        dval = val.decode('utf-8')
        print("GET PREFIX: {}".format(dval))
        return dval
    return val

CONFIG_PATH = "config.json"

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

MAX_RECORDING_COUNT = 30


class filepaths:
    # prefix = config['filepaths_prefix']
    # @property
    def prefix():
        return get_prefix()
    target_prefix = config['filepaths_prefix']
    # prefix = "./test_record/"
    hid_record = "{}hid_record.jsonl".format(prefix)
    audio_record = "{}audio_record.wav".format(prefix)
    video_record = "{}video_record.mp4".format(prefix)
    video_record_script = "{}video_record_script.sh".format(prefix)

    video_timestamps = "{}video_timestamps.json".format(prefix)
    hid_timestamps = "{}hid_timestamps.json".format(prefix)
    audio_timestamps = "{}audio_timestamps.json".format(prefix)


lock_key = "HID_MAIN_RECORDER_LOCK"
sig_on = "ON"
sig_off = "OFF"

timestep = 0.03
# filePath = "states.jsonl"


def set_redis_off():
    r = get_redis_client()
    r.set(lock_key, sig_off)


def set_redis_on():
    r = get_redis_client()
    r.set(lock_key, sig_on)


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


class TimestampedLogCreater:
    def __init__(self, file_name, indent_output=True):
        self.file_name = file_name
        self.timestamp_list = []
        self.indent_output = indent_output
        self.last_int_timestamp = -1

    def clear(self):
        self.timestamp_list = []

    def commit(self):
        timestamp = time.time()
        # show info every 1 second.
        int_timestamp = int(timestamp)
        if int_timestamp > self.last_int_timestamp:
            print("Appending timestamp at `{}`:".format(
                self.file_name), timestamp)
            self.last_int_timestamp = int_timestamp
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
                print("REMOVING OLD TIMESTAMPED CONTEXT AT: {}".format(
                    self.file_name))
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


def set_redis_off_on_exception():
    def exception_hook(exc_type, exc_value, tb):
        # print('Traceback:')
        # filename = tb.tb_frame.f_code.co_filename
        # name = tb.tb_frame.f_code.co_name
        # line_no = tb.tb_lineno
        # print(f"File {filename} line {line_no}, in {name}")

        # # Exception type and value
        # print(f"{exc_type.__name__}, Message: {exc_value}")
        if os.path.exists(get_prefix()):
            os.system("rm -rf {}".format(get_prefix()))
        set_redis_off()

        import traceback
        print("*** Traceback: ***")
        # traceback.print_tb(tb)
        # breakpoint()
        traceback.print_last()
        # traceback.print_exc()
        # traceback.print_tb(tb, limit=10)
        # print(dir(tb))
        # tb.print_tb()
        # traceback.print_exc()
        # tb.print_exc_info()

    sys.excepthook = exception_hook
