import logging
import schedule

# ft = logging.Filter("myfilter") # default filter is just a string checker

class MyFilter:
    @staticmethod
    def filter(record:logging.LogRecord):
        accepted=False
        msg = record.message
        print("MSG IN FILTER?", msg)
        return accepted

allow_logging = True