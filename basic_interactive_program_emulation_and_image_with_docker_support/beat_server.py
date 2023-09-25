import fastapi

app = fastapi.FastAPI()
import datetime
from beat_common import beat_server_address, beat_client_data

import pytz

# with respect to our dearly Py3.6
timezone_str = "Asia/Shanghai"
# timezone = pytz.timezone(timezone_str:='Asia/Shanghai')
timezone = pytz.timezone(timezone_str)
import schedule
import threading

from typing import Literal


def get_now_and_timestamp():
    now = get_now()
    timestamp = now.timestamp()
    return now, timestamp


UUID_TO_TIMESTAMP = {}
UUID_TO_STATUS = {} # dead -> True; alive -> False
ALIVE_THREADHOLD = 30

@app.get(beat_server_address["beat_url"])
def beat_request(uuid: str, action: Literal["hello", "heartbeat"]):
    now, timestamp = get_now_and_timestamp()
    UUID_TO_TIMESTAMP[uuid] = timestamp
    strtime = now.strftime(r"%Y-%m-%d %H:%M:%S")
    if action == "hello":
        print(f"client {uuid} hello at: %s" % strtime)
    else:
        print(f"received heartbeat from {uuid} at time {strtime}")
    return {beat_client_data["access_time_key"]: strtime}


def get_now():
    now = datetime.datetime.now(tz=timezone)
    return now


def check_alive():
    _, now_timestamp = get_now_and_timestamp()
    for uuid, timestamp in UUID_TO_TIMESTAMP.items():
        dead=True
        life = -ALIVE_THREADHOLD
        if life >=0:
            dead=False
        UUID_TO_STATUS[uuid] = dead


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, **{k: beat_server_address[k] for k in ["host", "port"]})
