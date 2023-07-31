"""
To use 'managed' loggers, you must import 'logger' from this file and pass it to other code.
"""

# python version check
from typing import Union

import sys  # recommend: 3.11.2

MIN_PY_VERSION = (3, 8)
if sys.version_info < MIN_PY_VERSION:
    raise Exception(f"Please use Python {'.'.join(MIN_PY_VERSION)} and above.")

# TODO: use `code_checker.py` to insert `log_utils` dependency to every py file under this folder. except for this one!

import logging
import schedule

# ft = logging.Filter("myfilter") # default filter is just a string checker
allow_logging = True
allow_huge_logging = True
HUGE_MSG_THRESHOLD = 100


def refresh_logger_lock():
    global allow_logging
    allow_logging = True


def refresh_huge_logger_lock():
    global allow_huge_logging
    allow_huge_logging = True


schedule.every(0.3).seconds.do(refresh_logger_lock)
schedule.every(1).seconds.do(refresh_huge_logger_lock)


# class MessageLengthAndFrequencyFilter:


#     @staticmethod
def messageLengthAndFrequencyFilter(record: logging.LogRecord):
    # def filter(record: logging.LogRecord):
    global allow_logging, allow_huge_logging, HUGE_MSG_THRESHOLD
    schedule.run_pending()
    # logger_print(dir(record))
    accepted = False
    # msg = record.msg
    # shall you intercept the args...
    # args = record.args # tuple. let's reset it.
    # breakpoint()
    msg = record.msg = record.msg % record.args
    args = record.args = ()

    if len(msg) < HUGE_MSG_THRESHOLD:
        if allow_logging:  # then this is some short message.
            accepted = True
            allow_logging = False
    else:
        if allow_huge_logging:
            record.msg = " ".join([msg[:HUGE_MSG_THRESHOLD], "..."])
            accepted = True
            allow_huge_logging = False
    return accepted


from logging import StreamHandler
import sys
import os

log_dir = os.path.join(os.path.dirname(__file__), "logs")

if os.path.exists(log_dir):
    if not os.path.isdir(log_dir):
        raise Exception(
            f"Non-directory object taking place of log directory `{log_dir}`."
        )
else:
    os.mkdir(log_dir)

log_filename = os.path.join(log_dir, "debug.log")

from logging.handlers import RotatingFileHandler

myHandler = RotatingFileHandler(
    log_filename, maxBytes=1024 * 1024 * 15, backupCount=3, encoding="utf-8"
)
myHandler.setLevel(logging.DEBUG)
# myHandler.setLevel(logging.INFO) # will it log less things? yes.
FORMAT = (  # add timestamp.
    "%(asctime)s <%(name)s:%(levelname)s> [%(pathname)s:%(lineno)s - %(funcName)s()]\n%(message)s"  # miliseconds already included!
    # "%(asctime)s.%(msecs)03d <%(name)s:%(levelname)s> [%(pathname)s:%(lineno)s - %(funcName)s()]\n%(message)s"
    # "<%(name)s:%(levelname)s> [%(pathname)s:%(lineno)s - %(funcName)s()]\n%(message)s"
)
# FORMAT = "<%(name)s:%(levelname)s> [%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
myFormatter = logging.Formatter(fmt=FORMAT)
myHandler.setFormatter(myFormatter)

stdout_handler = StreamHandler(sys.stdout)  # test with this!
stdout_handler.setLevel(logging.DEBUG)
# stdout_handler.addFilter(MessageLengthAndFrequencyFilter)
stdout_handler.addFilter(messageLengthAndFrequencyFilter)  # method also works!
stdout_handler.setFormatter(myFormatter)
# do not use default logger!
# logger = logging.getLogger(__name__)
logger = logging.getLogger("agi_computer_control")
logger.setLevel("DEBUG")
logger.addHandler(stdout_handler)
logger.addHandler(myHandler)

from rich.pretty import pretty_repr


def logger_print(*args):
    if len(args) != 0:
        format_string = "\n\n".join(["%s"] * len(args))
        # python 3.8+ required!
        logger.debug(
            format_string,
            *[
                pretty_repr(arg) if not isinstance(arg, Union[bytes, str]) else arg
                for arg in args
            ],
            stacklevel=2,
        )  # it is been called elsewhere.
        # logger.debug(
        #     "\n\n".join([pretty_repr(arg) if not isinstance(arg, Union[bytes, str]) else arg for arg in args]), stacklevel=2
        # )  # it is been called elsewhere.


import datetime

logger_print(
    f"[START LOGGING AT: {datetime.datetime.now().isoformat()}]".center(
        os.get_terminal_size().columns, "+"
    )
    # f"[START LOGGING AT: {datetime.datetime.now().isoformat()}]".center(70 - 2, "+")
)
# logging.basicConfig(
#     # filename=filename,
#     # level=logging.getLogger().getEffectiveLevel(),
#     level="DEBUG",
#     # stream=sys.stderr
#     force=True, # overridding root logger, which is deprecated.
#     handlers=[stdout_handler],
# )

if __name__ == "__main__":  # just a test.
    import time

    for i in range(100):
        time.sleep(0.1)
        logger.debug(f"test debug message {i}")
        logger.debug(f"test debug message {i} %s", "myarg")
        logger.debug(f"test huge message {i} " * 100)  # huge mssage
