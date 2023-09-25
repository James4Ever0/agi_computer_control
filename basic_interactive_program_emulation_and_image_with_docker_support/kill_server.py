# kill dead process by pid and remove them from beat server
# the kill server must emit beat signals, and can kill other processes to prove its effectiveness
# maybe we need to elevate
# import elevate
from beat_common import *

def kill_dead_process():
    ...

def kill_server_beat():
    ...

if __name__ == "__main__":
    while True:
        kill_server_beat()
        kill_dead_process()