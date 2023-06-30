import logging
import sys

logging.warning("abc") # default level: warning.

# logging.basicConfig(level=logging.INFO, stream=sys.stdout) # suppress debug output.
# level priority: DEBUG < INFO < WARNING < ERROR < CRITICAL
# logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
print("def")
logging.debug("abc")
print('2 abc')

# cannot override?
# logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, force=True) # force overriding. you can set it somewhere.
logging.debug("abc")
