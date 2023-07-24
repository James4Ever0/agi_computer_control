import logging
import schedule

# ft = logging.Filter("myfilter") # default filter is just a string checker
allow_logging = True


def refresh_logger_lock():
    global allow_logging
    allow_logging = True


schedule.every(1).seconds.do(refresh_logger_lock)


# class MessageLengthAndFrequencyFilter:
    
#     @staticmethod
def messageLengthAndFrequencyFilter(record: logging.LogRecord):
    # def filter(record: logging.LogRecord):
    global allow_logging
    schedule.run_pending()
    # print(dir(record))
    accepted = False
    if allow_logging:  # only if accepted we assign False to this variable.
        msg = record.msg
        # print("MSG IN FILTER?", msg)
        if len(msg) < 100:
            # print("ACCEPTED")
            accepted = True
            allow_logging = False
    return accepted


from logging import StreamHandler
import sys

stdout_handler = StreamHandler(sys.stdout)  # test with this!
stdout_handler.setLevel(logging.DEBUG)
# stdout_handler.addFilter(MessageLengthAndFrequencyFilter)
stdout_handler.addFilter(messageLengthAndFrequencyFilter) # method also works!
# do not use default logger!
logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
logger.addHandler(stdout_handler)
# logging.basicConfig(
#     # filename=filename,
#     # level=logging.getLogger().getEffectiveLevel(),
#     level="DEBUG",
#     # stream=sys.stderr
#     force=True, # overridding root logger, which is deprecated.
#     handlers=[stdout_handler],
# )

import time
for i in range(100):
    time.sleep(0.1)
    logger.debug(f"test debug message {i}")