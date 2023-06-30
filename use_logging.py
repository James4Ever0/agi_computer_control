import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout) # suppress debug output.
# DEBUG < INFO < WARNING < ERROR < CRITICAL
# logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
print("def")
logging.debug("abc")
print('2 abc')