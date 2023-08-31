# import twisted
# this could be used as test case.
import os
import traceback
from vocabulary import NaiveVocab
from cmath import nan

from type_utils import *

def safeDiv(a,b):
    """
    Return a/b if no exception is raised, otherwise nan.
    """
    ret = nan
    try:
        ret = a/b
    except ZeroDivisionError:
        pass
    return ret

def leftAndRightSafeDiv(a,b):
    """
    Return a/b and b/a, in safe division manner.
    """
    left_div = safeDiv(a,b)
    right_div = safeDiv(b,a)
    return left_div, right_div

if os.name == "nt":
    import wexpect as pexpect
    expected_wexpect_version = "4.0.0"
    wexp_version =  pexpect.__version__
    assert wexp_version == expected_wexpect_version, "wexpected version should be: {}\ncurrently: {}".format(expected_wexpect_version, wexp_version)
    import wexpect.host as host
    import socket


    def spawnsocket_read_nonblocking(self, size=1):
        """This reads at most size characters from the child application. If
        the end of file is read then an EOF exception will be raised.

        This is not effected by the 'size' parameter, so if you call
        read_nonblocking(size=100, timeout=30) and only one character is
        available right away then one character will be returned immediately.
        It will not wait for 30 seconds for another 99 characters to come in.

        This is a wrapper around Wtty.read()."""

        logger = host.logger
        EOF_CHAR = host.EOF_CHAR
        EOF = host.EOF

        if self.closed:
            logger.info("I/O operation on closed file in read_nonblocking().")
            raise ValueError("I/O operation on closed file in read_nonblocking().")

        try:
            s = self.sock.recv(size)

            if s:
                logger.debug(f"Readed: {s}")
            else:
                logger.spam(f"Readed: {s}")

            if EOF_CHAR in s:
                self.flag_eof = True
                logger.info("EOF: EOF character has been arrived")
                s = s.split(EOF_CHAR)[0]

        except ConnectionResetError:
            self.flag_eof = True
            logger.info("EOF('ConnectionResetError')")
            raise EOF("ConnectionResetError")
        except socket.timeout:
            return b""

    def spawnpipe_read_nonblocking(self, size=1):
        """This reads at most size characters from the child application. If
        the end of file is read then an EOF exception will be raised.

        This is not effected by the 'size' parameter, so if you call
        read_nonblocking(size=100, timeout=30) and only one character is
        available right away then one character will be returned immediately.
        It will not wait for 30 seconds for another 99 characters to come in.

        This is a wrapper around Wtty.read()."""

        logger = host.logger
        EOF_CHAR = host.EOF_CHAR
        EOF = host.EOF

        if self.closed:
            logger.warning("I/O operation on closed file in read_nonblocking().")
            raise ValueError("I/O operation on closed file in read_nonblocking().")

        try:
            s = host.win32file.ReadFile(self.pipe, size)[1]

            if s:
                logger.debug(f"Readed: {s}")
            else:
                logger.spam(f"Readed: {s}")

            if EOF_CHAR in s:
                self.flag_eof = True
                logger.info("EOF: EOF character has been arrived")
                s = s.split(EOF_CHAR)[0]

            return s
            # return s.decode()
        except host.pywintypes.error as e:
            if e.args[0] == host.winerror.ERROR_BROKEN_PIPE:  # 109
                self.flag_eof = True
                logger.info("EOF('broken pipe, bye bye')")
                raise EOF("broken pipe, bye bye")
            elif e.args[0] == host.winerror.ERROR_NO_DATA:
                """232 (0xE8): The pipe is being closed."""
                self.flag_eof = True
                logger.info("EOF('The pipe is being closed.')")
                raise EOF("The pipe is being closed.")
            else:
                raise

    host.SpawnSocket.read_nonblocking = spawnsocket_read_nonblocking
    host.SpawnPipe.read_nonblocking = spawnpipe_read_nonblocking

    def spawnbase_sendline(self, s=b""):
        s = enforce_bytes(s)
        n = self.send(s+b"\r\n")
        return n

    host.SpawnBase.sendline = spawnbase_sendline

else:
    import pexpect

    expected_pexpect_version = "4.6.0"
    pexp_version =  pexpect.__version__
    assert pexp_version == expected_pexpect_version, "pexpected version should be: {}\ncurrently: {}".format(expected_pexpect_version, pexp_version)

    def spawn_sendline(self, s=b""):
        s = enforce_bytes(s)
        return self.send(s + bytes(os.linesep))

    pexpect.spawn.sendline = spawn_sendline

    def spawnbase_read_nonblocking(self, size=1, timeout=None):
        """This reads data from the file descriptor.

        This is a simple implementation suitable for a regular file. Subclasses using ptys or pipes should override it.

        The timeout parameter is ignored.
        """

        try:
            s = os.read(self.child_fd, size)
        except OSError as err:
            if err.args[0] == errno.EIO:
                # Linux-style EOF
                self.flag_eof = True
                raise pexpect.spawnbase.EOF('End Of File (EOF). Exception style platform.')
            raise
        if s == b'':
            # BSD-style EOF
            self.flag_eof = True
            raise pexpect.spawnbase.EOF('End Of File (EOF). Empty string style platform.')

        # s = self._decoder.decode(s, final=False)
        self._log(s, 'read')
        return s

    pexpect.spawnbase.SpawnBase.read_nonblocking = spawnbase_read_nonblocking

# https://code.activestate.com/recipes/440554/
# wxpython, wexpect/winpexpect, pexpect
# https://peps.python.org/pep-3145/
# https://peps.python.org/pep-3156/
import sys
from collections import deque


def get_repr(content):
    if isinstance(content, str):
        content = content.encode()
    repr_content = content.hex()
    len_content = len(repr_content) / 2
    cut_len = 10
    return f"[{len_content}\tbyte{'s' if len_content != 0 else ''}] {repr_content[:cut_len*2]}{'...' if len_content > cut_len else ''}"


import time
import pytz
import datetime


timezone_str = "Asia/Shanghai"
timezone = pytz.timezone(timezone_str)


def formatTimeAtShanghai(timestamp):
    dt = datetime.datetime.fromtimestamp(timestamp, tz=timezone)
    return dt.isoformat()


from entropy_utils import ContentEntropyCalculator


class NaiveActor:
    def __init__(self, cmd):
        self.process = self.spawn(cmd)
        self.timeout = 1
        self.read_bytes = 0
        self.write_bytes = 0
        self.loop_count = 0
        self.start_time = time.time()
        self.read_head_bytes = 200
        self.read_tail_bytes = 200
        self.read_entropy_calc = ContentEntropyCalculator()
        self.write_entropy_calc = ContentEntropyCalculator()

    def spawn(self, cmd):
        return pexpect.spawn(cmd)
        # return pexpect.spawn(cmd, interact=True)  # will display

    def write(self, content):
        # if isinstance(content, bytes):
        #     content = content.decode()
        content = enforce_bytes(content)
        print("write:", get_repr(content), sep="\t")
        self.write_bytes += len(content)

        self.process.sendline(content)
        self.write_entropy_calc.count(content)
        return content

    def read(self):
        # cannot read.
        head_content = b""
        tail_content = deque([], maxlen=self.read_tail_bytes)
        read_byte_len = 0

        while True:
            try:
                kwargs = {}
                if os.name == "posix":
                    kwargs["timeout"] = self.timeout
                char = self.process.read_nonblocking(1, **kwargs)
                # print('char:', char)
                if isinstance(char, str):
                    char = char.encode()
                if char == b"":
                    break
                read_byte_len += 1
                self.read_entropy_calc.count(char)
                if len(head_content) < self.read_head_bytes:
                    head_content += char
                else:
                    tail_content.append(char)
            except:
                traceback.print_exc()
                break
        tail_content = b"".join(list(tail_content))
        if read_byte_len <= self.read_head_bytes + self.read_tail_bytes:
            sep = b""
        else:
            sep = b"\n...\n"
        content = sep.join([head_content, tail_content])

        print("read:", get_repr(content), sep="\t")
        self.read_bytes += read_byte_len
        return content

    def __del__(self):
        end_time = time.time()
        up_time = end_time - self.start_time
        read_ent = self.read_entropy_calc.entropy
        write_ent = self.write_entropy_calc.entropy
        rw_ratio, wr_ratio = leftAndRightSafeDiv(self.read_bytes, self.write_bytes)
        rw_ent_ratio, wr_ent_ratio = leftAndRightSafeDiv(read_ent, write_ent)
        print("summary".center(50, "="))
        print("start time:", formatTimeAtShanghai(self.start_time), sep="\t")
        print("end time:", formatTimeAtShanghai(end_time), sep="\t")
        print("up time:", up_time, sep="\t")
        print("loop count:", self.loop_count, sep="\t")
        print("total bytes read:", self.read_bytes, sep="\t")
        print("total bytes write:", self.write_bytes, sep="\t")
        print("r/w ratio:", rw_ratio)
        print("w/r ratio:", wr_ratio)
        print("read bytes entropy:", read_ent)
        print("write bytes entropy:", write_ent)
        print('r/w entropy ratio:', rw_ent_ratio)
        print('w/r entropy ratio:', wr_ent_ratio)

    def loop(self):
        self.read()
        self.write(NaiveVocab.generate())
        return True

    def run(self):
        # while True:
        while self.loop():
            print(f"[loop\t{str(self.loop_count)}]".center(60, "-"))
            self.loop_count += 1


if __name__ == "__main__":
    actor = NaiveActor(f"{sys.executable} naive_interactive.py")
    actor.run()
