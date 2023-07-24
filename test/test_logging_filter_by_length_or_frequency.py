import logging
import schedule

# ft = logging.Filter("myfilter") # default filter is just a string checker
allow_logging = True


def refresh_logger_lock():
    global allow_logging
    allow_logging = True


schedule.every(3).seconds.do(refresh_logger_lock)


class MessageLengthAndFrequencyFilter:
    @staticmethod
    def filter(record: logging.LogRecord):
        schedule.run_pending()
        accepted = False
        if allow_logging:
            msg = record.message
            print("MSG IN FILTER?", msg)
        return accepted

