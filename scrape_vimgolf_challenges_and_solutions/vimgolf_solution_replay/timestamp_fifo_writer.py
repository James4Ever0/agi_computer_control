#!/usr/bin/env python3
# not accurate. do not use in production
import atexit
import os

PIPE_PATH = "/tmp/timestamp_pipe"

if os.path.exists(PIPE_PATH):
    os.remove(PIPE_PATH)

os.mkfifo(PIPE_PATH)

atexit.register(lambda: os.remove(PIPE_PATH))

while True:
    # Block until a reader opens the pipe
    os.system("bash -c 'echo $(date +%s) > {}'".format(PIPE_PATH))
