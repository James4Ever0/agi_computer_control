import asyncio
import os

# import multitasking
# after all, we are single threaded
import time

from timeout_utils import *


@retrying_timeout_func(1, 11)  # passed
def os_sleep():
    print("running os sleep")
    os.system("sleep 10")
    os.system("echo hello world")
    print("exit os sleep")


@retrying_timeout_func(3, 2)  # not pass, fail attempts count: 3
def time_sleep():
    print("running time sleep")
    time.sleep(3)
    print("exit time sleep")


async def set_after(fut, delay, value):
    # Sleep for *delay* seconds.
    await asyncio.sleep(delay)

    # Set *value* as a result of *fut* Future.
    fut.set_result(value)


async def main():
    # Get the current event loop.
    loop = asyncio.get_running_loop()

    # Create a new Future object.
    fut = loop.create_future()

    # Run "set_after()" coroutine in a parallel Task.
    # We are using the low-level "loop.create_task()" API here because
    # we already have a reference to the event loop at hand.
    # Otherwise we could have just used "asyncio.create_task()".
    loop.create_task(set_after(fut, 1, "... world"))

    print("hello ...")

    # Wait until *fut* has a result (1 second) and print it.
    # print(await asyncio.wait_for(fut, timeout=1.5))
    print(await asyncio.wait_for(fut, timeout=0.5))


if __name__ == "__main__":
    os_sleep()
    time_sleep()
    # asyncio.run(main())
