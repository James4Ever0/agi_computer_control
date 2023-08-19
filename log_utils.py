"""
To use 'managed' loggers, you must import 'logger' from this file and pass it to other code.
"""

# TODO: configure file handlers for celery logging
# TODO: find a tool or make some script to take input from stdin and log & filter output

from rich.pretty import pretty_repr

def pretty(obj):
    return pretty_repr(obj)
# python version check

import sys  # recommend: 3.11.2

MIN_PY_VERSION = (3, 8)
if sys.version_info < MIN_PY_VERSION:
    logger_print = print
    logger_print(f"Please use Python {'.'.join([str(v) for v in MIN_PY_VERSION])} and above.")
else:
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
        setattr(record, "short_msg", msg)
        args = record.args = ()

        if len(msg) < HUGE_MSG_THRESHOLD:
            if allow_logging:  # then this is some short message.
                accepted = True
                allow_logging = False
        else:
            if allow_huge_logging:
                record.short_msg = " ".join([msg[:HUGE_MSG_THRESHOLD], "..."]) # do not put stdout in front of file handler!
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
    celery_log_filename = os.path.join(log_dir, "celery.log")
    fastapi_log_filename = os.path.join(log_dir, "fastapi.log")

    from logging.handlers import RotatingFileHandler




    import pytz
    # with respect to our dearly Py3.6
    timezone_str='Asia/Shanghai'
    # timezone = pytz.timezone(timezone_str:='Asia/Shanghai')
    timezone = pytz.timezone(timezone_str)
    # import logging
    import datetime

    FORMAT = (  # add timestamp.
        "%(asctime)s <%(name)s:%(levelname)s> [%(pathname)s:%(lineno)s - %(funcName)s()]\n%(message)s"  # miliseconds already included!
        # "%(asctime)s.%(msecs)03d <%(name)s:%(levelname)s> [%(pathname)s:%(lineno)s - %(funcName)s()]\n%(message)s"
        # "<%(name)s:%(levelname)s> [%(pathname)s:%(lineno)s - %(funcName)s()]\n%(message)s"
    )

    SHORT_FORMAT =  "%(asctime)s <%(name)s:%(levelname)s> [%(pathname)s:%(lineno)s - %(funcName)s()]\n%(short_msg)s"
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
                    s = dt.isoformat(timespec='milliseconds')
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
    logger = logging.getLogger("agi_computer_control")
    logger.setLevel("DEBUG")
    logger.addHandler(myHandler) # BUG: make sure long logs are unaffected in file.
    logger.addHandler(stdout_handler)



    def logger_print(*args, logger = logger):
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
                stacklevel=2,
            )  # it is been called elsewhere.
            # logger.debug(
            #     "\n\n".join([pretty_repr(arg) if not isinstance(arg, Union[bytes, str]) else arg for arg in args]), stacklevel=2
            # )  # it is been called elsewhere.


    import datetime

    try:
        terminal_column_size  = os.get_terminal_size().columns
    except:
        terminal_column_size = 30
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

    if __name__ == "__main__":  # just a test.
        import time

        for i in range(100):
            time.sleep(0.1)
            logger.debug(f"test debug message {i}")
            logger.debug(f"test debug message {i} %s", "myarg")
            logger.debug(f"test huge message {i} " * 100)  # huge mssage
