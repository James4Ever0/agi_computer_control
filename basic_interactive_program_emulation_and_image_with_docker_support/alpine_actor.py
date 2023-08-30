from naive_actor import NaiveActor
from vocabulary import AsciiVocab


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
import time

# import docker  # pip3 install docker
import progressbar
import func_timeout
import better_exceptions

# client = docker.from_env()
from logging.handlers import RotatingFileHandler

log_filename = "alpine_actor.log"
rthandler = RotatingFileHandler(
    log_filename, maxBytes=1024 * 1024 * 15, backupCount=3, encoding="utf-8"
)
# TODO: container & process profiler
import logging
import sys

logger = logging.getLogger("alpine_actor")

logger.setLevel(logging.DEBUG)
logger.addHandler(rthandler)
logger.addHandler(logging.StreamHandler(sys.stderr))

better_exceptions.SUPPORTS_COLOR = False


def log_and_print_unknown_exception():
    exc_type, exc_info, exc_tb = sys.exc_info()
    # traceback.print_exc()
    if exc_type is not None:
        exc_str = better_exceptions.format_exception(exc_type, exc_info, exc_tb)
        logger.debug(exc_str)
        print(exc_str)


# timeout this function.
# from functools import partial
import easyprocess
import os


def killAndPruneAllContainers():
    proc = easyprocess.EasyProcess("docker container ls").call()
    # proc = easyprocess.EasyProcess("docker container ls -a").call()
    if proc.stdout:
        lines = proc.stdout.split("\n")[1:]
        container_ids = [line.split(" ")[0] for line in lines]
        for cid in progressbar.progressbar(container_ids):
            cmd = f"docker container kill {cid}"
            try:
                func_timeout.func_timeout(2, os.system, args=(cmd,))
            except func_timeout.FunctionTimedOut:
                print(
                    f'timeout while killing container "{cid}".\nmaybe the container is not running.'
                )
            # os.system(f"docker container kill -s SIGKILL {cid}")
        os.system("docker container prune -f")


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

    def __del__(self):
        killAndPruneAllContainers()
        super().__del__()

    @staticmethod
    def timeit(func):
        def inner_func(self):
            start_time = time.time()
            # func(self)
            try:
                ret = func_timeout.func_timeout(self.max_loop_time, func, args=(self,))
            except func_timeout.FunctionTimedOut:
                print("Loop timeout %d exceeded." % self.max_loop_time)
                return
            end_time = time.time()
            rw_time = end_time - start_time
            print("rw time:", rw_time, sep="\t")
            if rw_time > self.max_rwtime:
                print("exit because of long rw time.\nmax rw time:", self.max_rwtime)
                return
            # return True
            return ret

        return inner_func

    @timeit
    def loop(self):
        _ = self.read()
        write_content = AsciiVocab.generate()
        write_content = write_content.encode()
        self.write(write_content)
        return True


# this will freeze the terminal. what the heck is going wrong?
# maybe we need to profile this program.
import gc

safe_exception_types = []
if os.name == "nt":
    import wexpect

    safe_exception_types.append(wexpect.wexpect_util.EOF)  # you can try to ignore this.
import traceback

def run_actor_forever(actor_class):
    # killAndPruneAllContainers()
    while True:
        actor = actor_class()
        try:
            actor.run()
        except KeyboardInterrupt:
            print("exit on user demand")
            break
        except Exception as e:
            check_if_is_safe_exception(e)
        del actor
        print()
        print("restarting actor")
        gc.collect()

def check_if_is_safe_exception(e):
    safe = False
    for exc_type in safe_exception_types:
        if isinstance(e, exc_type):
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
