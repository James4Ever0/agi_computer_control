import agg_python_bindings
import pty
import threading
# how to use pty/ptyproc:

# check pyxtermjs:
# https://pypi.org/project/pyxtermjs/
# https://github.com/cs01/pyxtermjs/blob/master/pyxtermjs/app.py

# check pyte examples:

# pexpect: https://pypi.org/project/pexpect/

import pty
import os
import subprocess
import select
import termios
import struct
import fcntl
import shlex
import logging
import time

__version__ = "0.5.0.2"

app_config = dict()
app_config["fd"] = None
app_config["child_pid"] = None
app_config['cmd'] = ['bash']


def set_winsize(fd, row, col, xpix=0, ypix=0):
    logging.debug("setting window size with termios")
    winsize = struct.pack("HHHH", row, col, xpix, ypix)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)


def read_and_forward_pty_output(
    callback, max_read_bytes=1024 * 20, poll_interval=0.01, errors="replace"
):
    while True:
        time.sleep(poll_interval)
        if app_config["fd"]:
            timeout_sec = 0
            (data_ready, _, _) = select.select([app_config["fd"]], [], [], timeout_sec)
            if data_ready:
                output = os.read(app_config["fd"], max_read_bytes).decode(
                    encoding="utf-8", errors=errors
                )
                callback(output)


def pty_input(data:str):
    """write to the child pty. The pty sees this as if you are typing in a real
    terminal.
    """
    if app_config["fd"]:
        logging.debug("received input: %s" % data)
        os.write(app_config["fd"], data.encode(encoding='utf-8'))


def resize(rows:int, cols:int):
    if app_config["fd"]:
        logging.debug(f"Resizing window to {rows}x{cols}")
        set_winsize(app_config["fd"], rows, cols)


def connect():
    """new client connected"""
    logging.info("new client connected")
    if app_config["child_pid"]:
        # already started child process, don't start another
        return

    # create child process attached to a pty we can read from and write to
    (child_pid, fd) = pty.fork()
    if child_pid == 0:
        # this is the child process fork.
        # anything printed here will show up in the pty, including the output
        # of this subprocess
        subprocess.run(app_config["cmd"])
    else:
        # this is the parent process fork.
        # store child fd and pid
        app_config["fd"] = fd
        app_config["child_pid"] = child_pid
        set_winsize(fd, 50, 50)
        cmd = " ".join(shlex.quote(c) for c in app_config["cmd"])
        # logging/print statements must go after this because... I have no idea why
        # but if they come before the background task never starts
        threading.Thread(target=read_and_forward_pty_output, daemon=True).start()

        logging.info("child pid is %s" % child_pid)
        logging.info(
            f"starting background task with command `{cmd}` to continously read "
            "and forward pty output to client"
        )
        logging.info("task started")


def main():
    ...

if __name__ == "__main__":
    main()
