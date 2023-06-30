import logging
import sys

# can format arbitrary object into string.

logging.warning("abc %s", (1,2)) # default level: warning.

# can only set once, unless using "force" to override.

# logging.basicConfig(level=logging.INFO, stream=sys.stdout) # suppress debug output.
# level priority: DEBUG < INFO < WARNING < ERROR < CRITICAL
# logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
print("def")
logging.debug("abc")
print('2 abc')

# cannot override?
# logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, force=True) # force overriding. you can set it somewhere.
logging.debug("abc", (1,2))
