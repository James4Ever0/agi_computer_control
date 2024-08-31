# count for io stats
# from remote_socket import SocketClient
import pexpect
import time
import threading
import numpy as np
from typing import Callable

# from eventemitter import EventEmitter # alternative: pyventus
# ref: https://github.com/asyncdef/eventemitter

from pyventus import AsyncIOEventEmitter
from strenum import StrEnum


class ProcessEvent(StrEnum):
    active = "ProcessActive"
    idle = "ProcessIdle"
    output = "ProcessOutput"
    exit = "ProcessExit"


class EventManager:
    def __init__(self):
        self.emitter = AsyncIOEventEmitter()
        self.event_linker = self.emitter._event_linker

    def on(self, event_name: str, callback: Callable):
        self.event_linker.on(event_name)(callback)

    def emit_classic(self, event, *args, **kwargs):
        self.emitter.emit(event, *args, **kwargs)

    def emit(self, event: str, *args, **kwargs):
        args = (event, *args)
        self.emit_classic(event, *args, **kwargs)


class ProcessEventWatcher:
    def __init__(
        self,
        command: str,
        speed_intervals: list[int],
        idle_threshold: int,
        watch_interval=1,
    ):
        self.watch_interval = watch_interval
        self.command = command
        self.process = pexpect.spawn(command, timeout=None)
        self.speed_intervals = speed_intervals
        self.speed_intervals.sort()
        self.idle_threshold = self.calculate_idle_threshold(idle_threshold)
        self.event_manager = EventManager()
        self.add_activity_event_listeners()
        self.stats = {
            "count": 0,
            "bytes": b"",
            "io_speed": {f"{it}s": 0 for it in speed_intervals},
        }
        self.datapoints = []
        self.process_idle = False
        self.maxpoints = max(speed_intervals) + 1

    def get_process_info(self):
        info = (
            f"PID: {self.process.pid} Command: {self.command} Idle: {self.process_idle}"
        )
        return info

    def activity_callback(self, name):
        print("[*]", self.get_process_info(), f"Event: {name}")

    def add_activity_event_listeners(self):
        self.on_idle(self.activity_callback)
        self.on_active(self.activity_callback)

    def calculate_idle_threshold(self, idle_threshold: int):
        for it in self.speed_intervals:
            if it >= idle_threshold:
                print(f"[*] Calculated idle threshold: {it} sec(s)")
                return it
        raise Exception(
            f"Unable to find a suitable idle threshold {idle_threshold} within speed intervals {self.speed_intervals}"
        )

    def update_stats(self):
        count = 0
        while self.process.isalive():
            one_byte = self.process.read(1)
            count += 1
            self.stats["count"] = count
            self.stats["bytes"] += one_byte

    def calculate_nth_average_speed(self, nth: int):
        nth = min(len(self.datapoints), nth)
        diff = np.diff(self.datapoints)
        ret = sum(diff[:nth]) / nth
        ret = float(-ret)
        return ret

    def wait_for_process_state(self, idle:bool, timeout:float, confirmation_threshold=1, loop_interval=1):
        if idle:
            inverse_state = False
        else:
            inverse_state = True
        elapsed_time = 0
        confirmation_count = 0
        while True:
            if self.process_idle == idle:
                confirmation_count += 1
            else:
                confirmation_count = 0
            if confirmation_count >= confirmation_threshold:
                break
            if elaspsed_time >= timeout:
                raise Exception(f"[-] Failed to wait for process state (idle: {idle}) within {timeout} sec(s) timeout limit")
            time.sleep(loop_interval)
            elapsed_time += loop_interval

    def wait_for_idle_state(self, timeout:float):
        self.wait_for_process_state(True, timeout)
    
    def wait_for_active_state(self, timeout:float):
        self.wait_for_process_state(False, timeout)

    def on_idle(self, callback: Callable):
        self.event_manager.on(ProcessEvent.idle, callback)

    def on_active(self, callback: Callable):
        self.event_manager.on(ProcessEvent.active, callback)

    def on_output(self, callback: Callable):
        self.event_manager.on(ProcessEvent.output, callback)

    def on_exit(self, callback: Callable):
        self.event_manager.on(ProcessEvent.exit, callback)

    def handle_process_output(self):
        read_bytes = self.stats.get("bytes")
        self.stats["bytes"] = b""  # clear the clutter
        self.event_manager.emit(ProcessEvent.output, data=read_bytes)

    def update_datapoints(self):
        count = self.stats.get("count")

        self.datapoints.insert(0, count)
        if len(self.datapoints) > self.maxpoints:
            self.datapoints = self.datapoints[: self.maxpoints]

    def update_speed(self):
        for it in self.speed_intervals:
            speed = self.calculate_nth_average_speed(it)
            self.stats["io_speed"][f"{it}s"] = speed

    def watch_once(self):
        self.handle_process_output()
        self.update_datapoints()
        self.update_speed()
        self.update_process_idle_state()

    def update_process_idle_state(self):
        idle_threshold_speed = self.stats["io_speed"][f"{self.idle_threshold}s"]

        if idle_threshold_speed > 0:
            if self.process_idle:
                self.process_idle = False
                self.emit(ProcessEvent.active)
        else:
            if not self.process_idle:
                self.process_idle = True
                self.emit(ProcessEvent.idle)

    def watch(self):
        self.start_daemon_thread(self.update_stats)
        while self.process.isalive():
            time.sleep(self.watch_interval)
            self.watch_once()

        self.handle_process_exit()

    def handle_process_exit(self):
        print("[*] Process exited with status code:", self.process.status)
        self.event_manager.emit(ProcessEvent.exit, status=self.process.status)

    def emit(self, event: ProcessEvent, *args, **kwargs):
        self.event_manager.emit(event, *args, **kwargs)

    def watch_in_background(self):
        self.start_daemon_thread(self.watch)

    @staticmethod
    def start_daemon_thread(target: Callable, *args, **kwargs):
        t = threading.Thread(target=target, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()


def naive_event_callback(name):
    print(f"[*] Event <{name}> received")


def second_naive_event_callback(name):
    print(f"[*] Event <{name}> received by the second listener")


# emitter.on('TerminalActive', naive_event_callback)
# emitter.on('TerminalIdle', naive_event_callback)
# or you can implement it yourself.


TMUX_SESSION_NAME = "test_tmux"


def main():
    command = f"tmux attach -t {TMUX_SESSION_NAME}"
    idle_threshold = 3
    speed_intervals = [1, 3, 5, 10]
    watcher = ProcessEventWatcher(command, speed_intervals, idle_threshold)

    watcher.on_idle(naive_event_callback)
    watcher.on_idle(second_naive_event_callback)

    watcher.on_active(naive_event_callback)
    watcher.on_active(second_naive_event_callback)

    watcher.watch()


if __name__ == "__main__":
    print("[*] Running terminal activity listener")
    main()
