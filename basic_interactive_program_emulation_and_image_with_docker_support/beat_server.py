import fastapi

# TODO: replace this with gui-attached panel & advanced rescue/countermeasures
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
UUID_TO_REGISTERED_TIMESTAMP = {}
UUID_TO_STATUS = {}  # alive -> True; dead -> False
ALIVE_THRESHOLD = 30


@app.get(beat_server_address["beat_url"])
def beat_request(uuid: str, action: Literal["hello", "heartbeat", "kill"]):
    # start = time.time()
    strtime, timestamp = get_strtime_and_timestamp()
    if action == "hello":
        print(f"client {uuid} hello at: %s" % strtime)
        UUID_TO_REGISTERED_TIMESTAMP[uuid] = timestamp
    elif action == "kill":
        print(f"client {uuid} killed at: %s" % strtime)
        for data_dict in [
            UUID_TO_REGISTERED_TIMESTAMP,
            UUID_TO_TIMESTAMP,
            UUID_TO_STATUS,
        ]:
            if uuid in data_dict.keys():
                del data_dict[uuid]
    elif action == "heartbeat":
        print(f"received heartbeat from {uuid} at time {strtime}")
    else:
        raise Exception(f"client {uuid} with unknown action:" + action)
    # end = time.time()
    if uuid not in UUID_TO_REGISTERED_TIMESTAMP.keys():
       print(f"client {uuid} not registered. registering.")
       UUID_TO_REGISTERED_TIMESTAMP[uuid] = timestamp
        # raise Exception(f"client {uuid} not registered.")
    UUID_TO_TIMESTAMP[uuid] = timestamp
    # print(f'request processing time: {end-start} secs', )
    return {beat_client_data["access_time_key"]: strtime}


def get_strtime_and_timestamp():
    now, timestamp = get_now_and_timestamp()
    strtime = now.strftime(r"%Y-%m-%d %H:%M:%S")
    return strtime, timestamp


def get_now():
    now = datetime.datetime.now(tz=timezone)
    return now


def check_alive():
    now_strtime, now_timestamp = get_strtime_and_timestamp()
    print(f"checking clients at {now_strtime}")
    for uuid, timestamp in UUID_TO_TIMESTAMP.items():
        registered_timestamp = UUID_TO_REGISTERED_TIMESTAMP[uuid]
        uptime = now_timestamp - registered_timestamp
        alive = True
        life = ALIVE_THRESHOLD - (now_timestamp - timestamp)
        if life < 0:
            alive = False
        UUID_TO_STATUS[uuid] = alive
        up_status = f"up: {uptime:.3f} secs"
        if alive:
            print(f"client {uuid} alive ({life:.3f} secs to death, {up_status})")
        else:
            print(f"client {uuid} ({up_status}) dead for {-life} seconds")
    status_list = UUID_TO_STATUS.values()
    print("summary".center(60, "="))
    print("total clients:", len(status_list))
    print("alive clients:", len([s for s in status_list if s == True]))
    print("dead clients:", len([s for s in status_list if s == False]))


import time

schedule.every(int(ALIVE_THRESHOLD / 3)).seconds.do(check_alive)


def check_alive_thread():
    while True:
        time.sleep(1)
        schedule.run_pending()


if __name__ == "__main__":
    import uvicorn

    thread = threading.Thread(target=check_alive_thread, daemon=True)
    thread.start()
    uvicorn.run(app, **{k: beat_server_address[k] for k in ["host", "port"]})
