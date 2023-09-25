# kill dead process by pid and remove them from beat server
# the kill server must emit beat signals, and can kill other processes to prove its effectiveness
# maybe we need to elevate
# import elevate
from beat_common import *
import os
import signal
import uuid
from log_common import *

killer_pid = os.getpid()
killer_uuid = str(uuid.uuid4())

def commit_kill(client_uuid: str, client_pid: int, client_role:str):
    kill_time = heartbeat_base_nocache(client_uuid, "kill", client_pid, client_role)
    print(f"commit kill for client {client_uuid} (pid: {client_pid}) at {kill_time}")


def kill_dead_process():
    dead_clients = []
    info = query_info()
    client_info= info['info']
    for client_uuid, client_info_dict in client_info.items():
        client_status = client_info_dict['status']
        client_pid = client_info_dict['pid']
        client_role = client_info_dict['role']
        if client_status is False:
            print(f"client {client_uuid} is dead.")
            dead_clients.append((client_uuid, client_pid, client_role))
    for client_uuid, client_pid, client_role in dead_clients:
        print("killing client:", client_uuid)
        try:
            os.kill(client_pid, signal.SIGKILL)
        except ProcessLookupError:
            print(f'client {client_uuid} (pid: {client_pid}) is already killed')
        except:
            log_and_print_unknown_exception()
            
        # remove from history.
        commit_kill(client_uuid, client_pid, client_role)


def kill_server_beat(action = 'heartbeat'):
    atime = heartbeat_base(killer_uuid, action, killer_pid, 'killer')
    print(f"killer {killer_uuid} beat at:", atime)

import time
if __name__ == "__main__":
    print(f"killer {killer_uuid} started (pid: {killer_pid})")
    kill_server_beat('hello')
    while True:
        kill_dead_process()
        time.sleep(1)
        kill_server_beat()
