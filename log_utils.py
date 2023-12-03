"""
To use 'managed' loggers, you must import 'logger' from this file and pass it to other code.
"""

# TODO: top-level exception hook (sys.excepthook)
# TODO: configure file handlers for celery logging
# TODO: find a tool or make some script to take input from stdin and log & filter output

from rich.pretty import pretty_repr


def pretty(obj):
    return pretty_repr(obj)


# python version check

import sys  # recommend: 3.11.2
import os

try:
    terminal_column_size = os.get_terminal_size().columns
except:
    terminal_column_size = 30

MIN_PY_VERSION = (3, 8)
SHOW_PYTHON_VERSION_WARNING = False

# TODO; make lower version of python supports logging utils
if sys.version_info < MIN_PY_VERSION:
    SHOW_PYTHON_VERSION_WARNING = True
    import inspect
    from exceptional_print import exprint

    FORMAT = "%(asctime)s <%(name)s:%(levelname)s> %(callerInfo)s\n%(message)s"

    SHORT_FORMAT = "%(asctime)s <%(name)s:%(levelname)s> %(callerInfo)s\n%(short_msg)s"

    def get_caller_info(level: int = 2):
        assert level >= 2, f"level {level} less than 2"
        caller_frame = inspect.currentframe().f_back
        for _ in range(level - 1):
            caller_frame = caller_frame.f_back
        # breakpoint()
        code_filename = caller_frame.f_code.co_filename
        code_relpath = os.path.relpath(code_filename)
        caller_info = "['%s:%s' - %s()]" % (
            code_relpath,
            caller_frame.f_lineno,
            caller_frame.f_code.co_name,
        )
        # exlogger_print(caller_info.center(60, "+"))
        # exlogger_print(*args, *[f"{k}:\t{v}" for k, v in kwargs.items()], sep=os.linesep)
        return caller_info

else:
    FORMAT = (  # add timestamp.
        "%(asctime)s <%(name)s:%(levelname)s> ['%(pathname)s:%(lineno)s' - %(funcName)s()]\n%(message)s"  # miliseconds already included!
        # "%(asctime)s.%(msecs)03d <%(name)s:%(levelname)s> [%(pathname)s:%(lineno)s - %(funcName)s()]\n%(message)s"
        # "<%(name)s:%(levelname)s> [%(pathname)s:%(lineno)s - %(funcName)s()]\n%(message)s"
    )

    SHORT_FORMAT = "%(asctime)s <%(name)s:%(levelname)s> ['%(pathname)s:%(lineno)s' - %(funcName)s()]\n%(short_msg)s"

# TODO: use `code_checker.py` to insert `log_utils` dependency to every py file under this folder. except for this one!

import logging
import schedule

# import traceback

# from exceptional_print import exprint as ep
import better_exceptions

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
    setattr(record, "short_msg", msg)
    args = record.args = ()

    if len(msg) < HUGE_MSG_THRESHOLD:
        if allow_logging:  # then this is some short message.
            accepted = True
            allow_logging = False
    else:
        if allow_huge_logging:
            record.short_msg = " ".join(
                [msg[:HUGE_MSG_THRESHOLD], "..."]
            )  # do not put stdout in front of file handler!
            accepted = True
            allow_huge_logging = False
    return accepted


from logging import StreamHandler

log_dir = os.path.join(os.path.dirname(__file__), "logs")

if os.path.exists(log_dir):
    if not os.path.isdir(log_dir):
        raise Exception(
            f"Non-directory object taking place of log directory `{log_dir}`."
        )
else:
    # os.system(f"mkdir -p {log_dir}")
    os.mkdir(log_dir)

log_filename = os.path.join(log_dir, "debug.log")
celery_log_filename = os.path.join(log_dir, "celery.log")
fastapi_log_filename = os.path.join(log_dir, "fastapi.log")

from logging.handlers import RotatingFileHandler

import pytz

# with respect to our dearly Py3.6
timezone_str = "Asia/Shanghai"
# timezone = pytz.timezone(timezone_str:='Asia/Shanghai')
timezone = pytz.timezone(timezone_str)
# import logging
import datetime


class Formatter(logging.Formatter):
    """override default 'logging.Formatter' to use timezone-aware datetime object"""

    def converter(self, timestamp):
        # Create datetime in UTC
        dt = datetime.datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        # Change datetime's timezone
        return dt.astimezone(timezone)

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec="milliseconds")
            except TypeError:
                s = dt.isoformat()
        return s


myFormatter = Formatter(fmt=FORMAT)
myShortFormatter = Formatter(fmt=SHORT_FORMAT)


def makeRotatingFileHandler(log_filename: str, level=logging.DEBUG):
    myHandler = RotatingFileHandler(
        log_filename, maxBytes=1024 * 1024 * 15, backupCount=3, encoding="utf-8"
    )
    myHandler.setLevel(level)
    myHandler.setFormatter(myFormatter)
    return myHandler


myHandler = makeRotatingFileHandler(log_filename)
# myHandler.setLevel(logging.INFO) # will it log less things? yes.

# FORMAT = "<%(name)s:%(levelname)s> [%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
# myFormatter = logging.Formatter(fmt=FORMAT)
# myHandler.setFormatter(myFormatter)

stdout_handler = StreamHandler(sys.stdout)  # test with this!
stdout_handler.setLevel(logging.DEBUG)
# stdout_handler.addFilter(MessageLengthAndFrequencyFilter)
stdout_handler.addFilter(messageLengthAndFrequencyFilter)  # method also works!
stdout_handler.setFormatter(myShortFormatter)
# do not use default logger!
# logger = logging.getLogger(__name__)
logger = logging.getLogger("microgrid")
logger.setLevel("DEBUG")
logger.addHandler(myHandler)  # BUG: make sure long logs are unaffected in file.
logger.addHandler(stdout_handler)


def logger_print(*args, logger=logger, stacklevel = 2):
    if len(args) != 0:
        format_string = "\n\n".join(["%s"] * len(args))
        # python 3.8+ required!
        logger.debug(
            format_string,
            *[
                # fallback for older versions:
                pretty(arg)
                if not any(isinstance(arg, t) for t in [bytes, str])
                else arg
                # pretty_repr(arg) if not isinstance(arg, Union[bytes, str]) else arg
                for arg in args
            ],
            **(
                {"stacklevel": stacklevel}
                if not SHOW_PYTHON_VERSION_WARNING
                else {"extra": {"callerInfo": get_caller_info(level = stacklevel)}}
            ),
        )  # it is been called elsewhere.
        # logger.debug(
        #     "\n\n".join([pretty_repr(arg) if not isinstance(arg, Union[bytes, str]) else arg for arg in args]), stacklevel=2
        # )  # it is been called elsewhere.


import datetime

logger_print(
    f"[START LOGGING AT: {datetime.datetime.now().isoformat()}]".center(
        terminal_column_size, "+"
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


def logger_excepthook(exc_type, exc_value, tb):
    
    with pretty_format_excinfo_context(exc_type, exc_value, tb) as formatted:
        formatted_exc = ["<TOPLEVEL EXCEPTION>", formatted]
        logger_print(*formatted_exc)
    better_exceptions.excepthook(exc_type, exc_value, tb)

from contextlib import contextmanager

@contextmanager
def pretty_format_excinfo_context(exc_type, exc_value, tb):
    try:
        better_exceptions.SUPPORTS_COLOR = False
        formatted = "".join(better_exceptions.format_exception(exc_type, exc_value, tb))
        yield formatted
    finally:
        better_exceptions.SUPPORTS_COLOR = True

def logger_traceback_print():
    with pretty_format_excinfo_context(*sys.exc_info()) as formatted:
        logger_print(formatted, stacklevel = 3)


sys.excepthook = logger_excepthook
logger_print("logging started at directory: ", os.path.abspath(os.curdir))
if SHOW_PYTHON_VERSION_WARNING:
    logger_print(
        f"Please use Python {'.'.join([str(v) for v in MIN_PY_VERSION])} and above."
    )
if __name__ == "__main__":  # just a test.
    import time

    for i in range(100):
        time.sleep(0.1)
        logger.debug(f"test debug message {i}")
        logger.debug(f"test debug message {i} %s", "myarg")
        logger.debug(f"test huge message {i} " * 100)  # huge mssage
