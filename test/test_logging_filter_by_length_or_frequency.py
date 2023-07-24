import logging
import schedule

# ft = logging.Filter("myfilter") # default filter is just a string checker
allow_logging = True

def 

schedule.every(3).seconds.do()

class MyFilter:
    @staticmethod
    def filter(record:logging.LogRecord):
        schedule.run_pending()
        accepted=False
        msg = record.message
        print("MSG IN FILTER?", msg)
        return accepted
