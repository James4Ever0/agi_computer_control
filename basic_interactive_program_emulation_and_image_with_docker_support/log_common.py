
import sys

import logging
from logging.handlers import RotatingFileHandler
import better_exceptions

log_filename = "actors.log"
rthandler = RotatingFileHandler(
    log_filename, maxBytes=1024 * 1024 * 15, backupCount=3, encoding="utf-8"
)

logger = logging.getLogger("actors")

logger.setLevel(logging.DEBUG)
logger.addHandler(rthandler)
logger.addHandler(logging.StreamHandler(sys.stderr))

better_exceptions.SUPPORTS_COLOR = False


def log_and_print_unknown_exception():
    exc_type, exc_info, exc_tb = sys.exc_info()
    # traceback.print_exc()
    if exc_type is not None:
        exc_str = "\n".join(
            better_exceptions.format_exception(exc_type, exc_info, exc_tb)
        )
        logger.debug(exc_str)
        print(exc_str)
