import logging
import sys

# can format arbitrary object into string.


def get_logging_level():
    logging_level = logging.getLogger().getEffectiveLevel()
    logging.critical("LOGGING LEVEL: %s" logging_level)
    return logging_level


logging.warning("abc %s", (1, 2))  # default level: warning.

# can only set once, unless using "force" to override.
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s:%(message)s",
    force=True,
)
# logging.basicConfig(level=logging.INFO, stream=sys.stdout) # suppress debug output.
# level priority: DEBUG < INFO < WARNING < ERROR < CRITICAL
# logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
print("def")
logging.debug("abc")
print("2 abc")

# cannot override?
# logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, force=True) # force overriding. you can set it somewhere.
logging.debug("abc", (1, 2))

logging.info("abc %s", {1: []})
