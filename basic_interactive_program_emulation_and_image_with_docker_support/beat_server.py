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

# import threading

from typing import Literal


def get_now_and_timestamp():
    now = get_now()
    timestamp = now.timestamp()
    return now, timestamp


UUID_TO_TIMESTAMP = {}
UUID_TO_REGISTERED_TIMESTAMP = {}
UUID_TO_STATUS = {}  # alive -> True; dead -> False
UUID_TO_PID = {}
UUID_TO_ROLE = {}
ALIVE_THRESHOLD = 30 * 2  # 30 is a bit of low.


@app.get(beat_server_address["info_url"])
def get_info():
    schedule.run_pending()
    _, timestamp = get_now_and_timestamp()
    return {
        "info": {
            k: {
                "status": v,
                "pid": UUID_TO_PID[k],
                "role": UUID_TO_ROLE[k],
                "timestamp": UUID_TO_TIMESTAMP[k],
            }
            for k, v in UUID_TO_STATUS.items()
        },
        "timestamp": timestamp,
    }


# TODO: delegate this kill signal to other process
# TODO: pass pid with uuid
# TODO: get unassigned uuid from here, instead of anywhere else
# TODO: distributed watchdog & recursive keep alive signal mechanism
# TODO: count revive time & frequencies
@app.get(beat_server_address["beat_url"])
def beat_request(
    uuid: str,
    action: Literal["hello", "heartbeat", "kill"],
    role: Literal["killer", "client", "server"],  # can also be server?
    pid: int,
):
    # start = time.time()
    strtime, timestamp = get_strtime_and_timestamp()
    if action != "kill":
        for data_dict, it, it_name in [
            (UUID_TO_PID, pid, "PID"),
            (UUID_TO_ROLE, role, "ROLE"),
        ]:
            if uuid not in data_dict.keys():
                data_dict[uuid] = it
            elif (old_it := data_dict[uuid]) != it:
                raise Exception(f"{it_name} mismatch! (old: {old_it}, new: {it})")
    if action == "hello":
        print(f"client {uuid} hello at:", strtime)
        UUID_TO_REGISTERED_TIMESTAMP[uuid] = timestamp
    elif action == "kill":
        print(f"client {uuid} killed at:", strtime)
        for data_dict in [
            UUID_TO_TIMESTAMP,
            UUID_TO_REGISTERED_TIMESTAMP,
            UUID_TO_STATUS,
            UUID_TO_PID,
            UUID_TO_ROLE,
        ]:
            if uuid in data_dict.keys():
                del data_dict[uuid]
    elif action == "heartbeat":
        print(f"received heartbeat from {uuid} at time {strtime}")
    else:
        raise Exception(f"client {uuid} with unknown action:" + action)
    # end = time.time()
    if action != "kill":
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
    alive_roles = []
    dead_roles = []
    print(f"checking clients at {now_strtime}")
    for uuid, registered_timestamp in UUID_TO_REGISTERED_TIMESTAMP.items():
        timestamp = UUID_TO_TIMESTAMP.get(uuid, registered_timestamp)
        role = UUID_TO_ROLE.get(uuid, "unknown")
        pid = UUID_TO_PID.get(uuid, -1)
        uptime = now_timestamp - registered_timestamp
        alive = True
        life = ALIVE_THRESHOLD - (now_timestamp - timestamp)
        if life < 0:
            alive = False
        UUID_TO_STATUS[uuid] = alive
        up_status = f"up: {uptime:.3f} secs"
        pid_info = f"pid: {pid}"
        if alive:
            print(
                f"client {uuid} alive ({pid_info}, {life:.3f} secs to death, {up_status})"
            )
            alive_roles.append(role)
        else:
            print(
                f"client {uuid} ({pid_info}, {up_status}) dead for {-life:.3f} seconds"
            )
            dead_roles.append(role)
    status_list = UUID_TO_STATUS.values()
    print("summary".center(60, "="))
    print("total clients:", len(status_list))
    print("alive clients:", *role_statistics(alive_roles))
    print("dead clients:", *role_statistics(dead_roles))


from typing import List


def role_statistics(roles: List[str]):
    details = ", ".join([f"{r}: {roles.count(r)}" for r in set(roles)])
    return len(roles), f"({details})" if details else ""


# import time

schedule.every(int(ALIVE_THRESHOLD / 3)).seconds.do(check_alive)


# def check_alive_thread():
#     while True:
#         time.sleep(1)
#         schedule.run_pending()


if __name__ == "__main__":
    import uvicorn

    # thread = threading.Thread(target=check_alive_thread, daemon=True)
    # thread.start()
    uvicorn.run(app, **{k: beat_server_address[k] for k in ["host", "port"]})
