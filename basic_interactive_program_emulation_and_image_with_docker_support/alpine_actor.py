# this will freeze the terminal. what the heck is going wrong?
# maybe we need to profile this program.
import gc
import getpass

# TODO: container & process profiler
import logging
import os
import sys
import traceback
import shutil

# client = docker.from_env()
from logging.handlers import RotatingFileHandler

import better_exceptions
import easyprocess
import elevate

# timeout this function.
# from functools import partial
import func_timeout

# import docker  # pip3 install docker
import progressbar

from naive_actor import NaiveActor
from vocabulary import AsciiVocab

REQUIRED_BINARIES = ['docker']

for name in REQUIRED_BINARIES:
    assert shutil.which(name), f"{name} is not available in PATH." # you can specify absolute path here

LEGACY_DOCKER = False
if sys.maxsize < 2**32:
    print("Your system is 32bit or lower.")
    print("Assume using legacy docker.")
    LEGACY_DOCKER = True
    if os.name == "posix":
        # check if is sudo
        print("*nix system detected.")
        # you don't need to do root checking
        # username = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))
        # username = getpass.getuser()
        # # ref: https://www.geeksforgeeks.org/how-to-get-the-current-username-in-python/
        # is_sudo = username == "root"
        # if not is_sudo:
        #     msg = f"You ({username}) are not sudo. Docker may malfunction."
        #     # raise Exception(msg)
        #     print(msg)
        #     print("Elevating now.")
        elevate.elevate(graphical=False)


class _AutoSeparatedString(str):
    __slots__ = ["sep"]

    # def __init__(self, *args, **kwargs):
    #     self.sep = kwargs.pop('sep')
    def __add__(self, other):
        s_self, s_other = str(self), str(other)
        val = s_self.__add__(self.sep + s_other)
        return self.__class__(val)

    def __radd__(self, other):
        s_self, s_other = str(self), str(other)
        val = s_other.__add__(self.sep + s_self)
        return self.__class__(val)


class AutoSpacedString(_AutoSeparatedString):
    sep = " "
    # def __init__(self, *args, **kwargs):
    #     kwargs['sep'] = ' '
    #     super().__init__(*args, **kwargs)


# a = AutoSpacedString('a')
# # a = AutoSeparatedString('a')
# print(a)
# print(a+a)
# print(a+a+a)
# you had better adopt async/await syntax.
# import time


log_filename = "alpine_actor.log"
rthandler = RotatingFileHandler(
    log_filename, maxBytes=1024 * 1024 * 15, backupCount=3, encoding="utf-8"
)

logger = logging.getLogger("alpine_actor")

logger.setLevel(logging.DEBUG)
logger.addHandler(rthandler)
logger.addHandler(logging.StreamHandler(sys.stderr))

better_exceptions.SUPPORTS_COLOR = False


def log_and_print_unknown_exception():
    exc_type, exc_info, exc_tb = sys.exc_info()
    # traceback.print_exc()
    if exc_type is not None:
        exc_str = "\n".join(
            better_exceptions.format_exception(exc_type, exc_info, exc_tb)
        )
        logger.debug(exc_str)
        print(exc_str)


def docker_cmd(*args):
    return " ".join(["docker", *args])


def docker_container_cmd(*args):
    return docker_cmd("container", *args)


if LEGACY_DOCKER:
    LIST_CONTAINER = docker_cmd("ps -a")
    KILL_CONTAINER = docker_cmd("rm -f")
    # KILL_CONTAINER = docker_cmd("kill")
else:
    LIST_CONTAINER = docker_container_cmd("ls")
    KILL_CONTAINER = docker_container_cmd("kill")

# this error has been recorded.
# we cannot just leave it like this.
# we need some watchdog thread.
# DOCKER CLI ON MACOS IS NOT RESPONSIVE!
# WHAT TO DO WITH THIS?
# DO NOT FOOL ME INTO BUYING E5-2650V2 OR REGECC RAMS!
# suggestion: use ssh-based interaction with containers.
# suggestion: restart docker service on macos.
# TODO: make unittests for failsafe protocols and watchdogs
# TODO: check docker binary if it is in PATH
# TODO: count failures of microtasks like this method and create remedy routines which trying to repair and continue execution

def killAndPruneAllContainers():
    fail_counter = 0
    for _ in range(3):
        try:
            success = _killAndPruneAllContainers()
            assert success, "Failed to execute docker kill and prune"
        except:
            fail_counter +=1

@func_timeout.func_set_timeout(timeout=10)
def _killAndPruneAllContainers():  # not working for legacy docker.
    success = False
    proc = easyprocess.EasyProcess(LIST_CONTAINER).call()
    # proc = easyprocess.EasyProcess("docker container ls -a").call()
    if proc.stdout:
        lines = proc.stdout.split("\n")[1:]
        container_ids = [line.split(" ")[0] for line in lines]
        for cid in progressbar.progressbar(container_ids):
            cmd = f"{KILL_CONTAINER} {cid}"
            try:
                func_timeout.func_timeout(2, os.system, args=(cmd,))
                success = True
            except func_timeout.FunctionTimedOut:
                print(
                    f'timeout while killing container "{cid}".\nmaybe the container is not running.'
                )
            # os.system(f"docker container kill -s SIGKILL {cid}")
        if not LEGACY_DOCKER:
            os.system("docker container prune -f")
    return success



# BUG: deprecated! may not connect to docker socket on windows.
# @partial(func_timeout.func_timeout, 10)
# def killAndPruneAllContainers():
#     # def stopAndPruneAllContainers():
#     all_containers = client.containers.list(all=True)
#     print("killing running containers...")

#     for container in progressbar.progressbar(all_containers):
#         try:
#             container.kill()
#         except:
#             log_and_print_unknown_exception()
#             # container not running. can be pruned.
#             # usually.
#         # container.stop()
#     print("pruning stopped containers...")
#     client.containers.prune()


class AlpineActor(NaiveActor):
    def __init__(self):
        self.max_rwtime = 0.5
        self.max_loop_time = 3
        killAndPruneAllContainers()
        super().__init__("docker run --rm -it alpine:3.7")
        # TODO: detect if the container is down by heartbeat-like mechanism
        # TODO: retrieve created container id

    def __del__(self):
        killAndPruneAllContainers()
        super().__del__()

    @NaiveActor.timeit
    def loop(self):
        _ = self.read()
        write_content = AsciiVocab.generate()
        write_content = write_content.encode()
        self.write(write_content)
        return True


SAFE_EXCEPTION_TYPES = []
if os.name == "nt":
    import wexpect

    SAFE_EXCEPTION_TYPES.append(wexpect.wexpect_util.EOF)  # you can try to ignore this.


# from typing import Generator
def run_actor_forever(actor_class):
    # killAndPruneAllContainers()
    if hasattr(actor_class, "__next__"):
        # if isinstance(actor_class, Generator):
        make_actor = lambda: next(actor_class)
    else:
        make_actor = lambda: actor_class()

    # we just cannot use such long timeout limit.
    # need watchdog alternative.
    @func_timeout.func_set_timeout(timeout=131)
    def internal_loop():
        ret = None
        # actor = actor_class()
        actor = make_actor()

        @func_timeout.func_set_timeout(timeout=100)
        def run_actor():
            try:
                actor.run()
            except KeyboardInterrupt:
                print("exit on user demand")
                return "INTERRUPTED"
            except Exception as e:
                check_if_is_safe_exception(e)

        ret = run_actor()
        del actor
        if ret is None:
            print()
            print("restarting actor")
        gc.collect()
        return ret

    while True:
        ret = None
        try:
            ret = internal_loop()
            if ret == "INTERRUPTED":
                break
        except Exception as e:
            check_if_is_safe_exception(e)


def check_if_is_safe_exception(e):
    safe = False
    # for exc_type in SAFE_EXCEPTION_TYPES:
    #     if isinstance(e, exc_type):
    if type(e) in SAFE_EXCEPTION_TYPES:
        safe = True
    if safe:
        traceback.print_exc()
        print("safe exception:", e)
    else:
        log_and_print_unknown_exception()
    return safe


if __name__ == "__main__":
    run_actor_forever(AlpineActor)
#     import cProfile
#     fpath = "alpine_actor.profile"
#     # # print("running")
#     prof = cProfile.run("AlpineActor()", filename=fpath)
#     # print("hello world")
