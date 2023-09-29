# import twisted
# this could be used as test case.
# TODO: survive reopening the laptop lid
# TODO: improve task execution logic, eliminate long running blocking tasks.
# TODO: use celery to schedule tasks
import datetime
from beat_common import heartbeat_base
import os
import sys
import time
import copy
import traceback
from cmath import nan
from log_common import *
import uuid
current_pid = os.getpid()
print("current_pid:", current_pid)
actor_uuid = str(uuid.uuid4())
strtime = heartbeat_base(uuid=actor_uuid, action = 'hello', pid=current_pid, role='client')
print('beat server hello: %s' % strtime)

class InteractiveChallengeFailed(Exception):
    """
    If "expect" like challenge failed for some reason, raise this exception.
    """

    ...


# https://code.activestate.com/recipes/440554/
# wxpython, wexpect/winpexpect, pexpect
# https://peps.python.org/pep-3145/
# https://peps.python.org/pep-3156/
from collections import deque

import func_timeout
import pytz
from pydantic import BaseModel

from entropy_utils import ContentEntropyCalculator
from type_utils import *
from vocabulary import NaiveVocab


def unicodebytes(string: str):
    return bytes(string, encoding="utf8")


class ActorStats(BaseModel):
    start_time: float
    end_time: float
    up_time: float
    loop_count: int
    read_bytes: int
    write_bytes: int
    read_ent: float
    write_ent: float
    rw_ratio: float
    wr_ratio: float
    rw_ent_ratio: float
    wr_ent_ratio: float


def safeDiv(a, b):
    """
    Return a/b if no exception is raised, otherwise nan.
    """
    ret = nan
    try:
        ret = a / b
    except ZeroDivisionError:
        pass
    return ret


def leftAndRightSafeDiv(a, b):
    """
    Return a/b and b/a, in safe division manner.
    """
    left_div = safeDiv(a, b)
    right_div = safeDiv(b, a)
    return left_div, right_div


READ_KNOWN_EXCEPTIONS = []
# SOCKET_TIMEOUT = .2
# SOCKET_TIMEOUT = .01
SOCKET_TIMEOUT = 0.001
from contextlib import contextmanager

if os.name == "nt":
    NT_CONTEXT = dict(NT_READ_NONBLOCKING_DECODE=False, NT_ENCODING="utf-8")

    @contextmanager
    def nt_read_nonblocking_decode_context():
        NT_CONTEXT["NT_READ_NONBLOCKING_DECODE"] = True
        try:
            yield
        finally:
            NT_CONTEXT["NT_READ_NONBLOCKING_DECODE"] = False

    import wexpect as pexpect

    expected_wexpect_version = "4.0.0"
    wexp_version = pexpect.__version__
    assert (
        wexp_version == expected_wexpect_version
    ), "wexpected version should be: {}\ncurrently: {}".format(
        expected_wexpect_version, wexp_version
    )
    import wexpect.host as host
    import socket

    def spawnsocket_connect_to_child(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.sock.settimeout(SOCKET_TIMEOUT)

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
            return "" if NT_CONTEXT["NT_READ_NONBLOCKING_DECODE"] else b""

        return (
            s.decode(NT_CONTEXT["NT_ENCODING"])
            if NT_CONTEXT["NT_READ_NONBLOCKING_DECODE"]
            else s
        )

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

            # return s
            return (
                s.decode(NT_CONTEXT["NT_ENCODING"])
                if NT_CONTEXT["NT_READ_NONBLOCKING_DECODE"]
                else s
            )
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

    host.SpawnSocket.connect_to_child = spawnsocket_connect_to_child
    host.SpawnSocket.read_nonblocking = spawnsocket_read_nonblocking
    host.SpawnPipe.read_nonblocking = spawnpipe_read_nonblocking

    def spawnbase_sendline(self, s=b""):
        s = enforce_bytes(s)
        n = self.send(s + b"\r\n")
        return n

    host.SpawnBase.sendline = spawnbase_sendline

else:
    import pexpect

    # let's skip version check, for kail.
    # expected_pexpect_version = "4.6.0"
    # pexp_version = pexpect.__version__
    # assert (
    #     pexp_version == expected_pexpect_version
    # ), "pexpected version should be: {}\ncurrently: {}".format(
    #     expected_pexpect_version, pexp_version
    # )

    READ_KNOWN_EXCEPTIONS.append(pexpect.pty_spawn.TIMEOUT)
    READ_KNOWN_EXCEPTIONS.append(pexpect.spawnbase.EOF)  # are you sure?

    def spawn_sendline(self, s=b""):
        s = enforce_bytes(s)
        return self.send(s + unicodebytes(os.linesep))

    pexpect.spawn.sendline = spawn_sendline

    def spawnbase_read_nonblocking(self, size=1, timeout=None):
        """This reads data from the file descriptor.

        This is a simple implementation suitable for a regular file. Subclasses using ptys or pipes should override it.

        The timeout parameter is ignored.
        """

        try:
            s = os.read(self.child_fd, size)
        except OSError as err:
            if err.args[0] == pexpect.spawnbase.errno.EIO:
                # Linux-style EOF
                self.flag_eof = True
                raise pexpect.spawnbase.EOF(
                    "End Of File (EOF). Exception style platform."
                )
            raise
        if s == b"":
            # BSD-style EOF
            self.flag_eof = True
            raise pexpect.spawnbase.EOF(
                "End Of File (EOF). Empty string style platform."
            )

        # s = self._decoder.decode(s, final=False)
        self._log(s, "read")
        return s

    pexpect.spawnbase.SpawnBase.read_nonblocking = spawnbase_read_nonblocking


def get_repr(content):
    if isinstance(content, str):
        content = content.encode()
    repr_content = content.hex()
    len_content = len(repr_content) / 2
    assert (
        len_content % 1 == 0.0
    ), f"possible counting mechanism failure\nnon-integral content length detected: {len_content}"
    len_content = int(len_content)
    cut_len = 10
    return f"[{len_content}\tbyte{'s' if len_content != 0 else ''}] {repr_content[:cut_len*2]}{'...' if len_content > cut_len else ''}"


timezone_str = "Asia/Shanghai"
timezone = pytz.timezone(timezone_str)


def formatTimeAtShanghai(timestamp):
    dt = datetime.datetime.fromtimestamp(timestamp, tz=timezone)
    return dt.isoformat()


class NaiveActor:
    write_method = lambda proc: proc.sendline
    actorStatsClass = ActorStats

    @staticmethod
    def timeit(func):
        def inner_func(self):
            start_time = time.time()
            # func(self)
            try:
                ret = func_timeout.func_timeout(self.max_loop_time, func, args=(self,))
            except func_timeout.FunctionTimedOut:
                print("Loop timeout %d exceeded." % self.max_loop_time)
                return
            finally:
                end_time = time.time()
                rw_time = end_time - start_time
                print("rw time:", rw_time, sep="\t")
                if rw_time > self.max_rwtime:
                    print(
                        "exit because of long rw time.\nmax rw time:", self.max_rwtime
                    )
                    return
            return ret

        return inner_func

    def __init__(self, cmd, encoding="utf-8"):
        self.process = self.spawn(cmd)
        self.encoding = encoding

        if os.name == 'nt':
            NT_CONTEXT['NT_ENCODING'] = encoding
            win_expect_old = copy.copy(self.process.expect)
            def win_expect_new(*args, **kwargs):
                with nt_read_nonblocking_decode_context():
                    return win_expect_old(*args, **kwargs)
            self.process.expect = win_expect_new
        self.timeout = SOCKET_TIMEOUT
        self.max_loop_time = 3
        self.max_init_time = 12
        self.max_rwtime = 0.5
        # self.timeout = 0.2 # equivalent to wexpect
        # self.timeout = 0.001
        # self.timeout = 1 # will cause havoc if set it too long
        self.read_bytes = 0
        self.write_bytes = 0
        self.loop_count = 0
        self.start_time = time.time()
        self.read_head_bytes = 200
        self.read_tail_bytes = 200
        self.read_entropy_calc = ContentEntropyCalculator()
        self.write_entropy_calc = ContentEntropyCalculator()
        self._stats = ...
        self.recent_loop_threshold = 300
        """
        To limiting history data size for calculating recent statistics.
        """

    def spawn(self, cmd):
        return pexpect.spawn(cmd)
        # return pexpect.spawn(cmd, interact=True)  # will display

    def write(self, content):
        # if isinstance(content, bytes):
        #     content = content.decode()
        content = enforce_bytes(content)
        print("write:", get_repr(content), sep="\t")
        self.write_bytes += len(content)

        write_method = self.__class__.write_method(self.process)
        write_method(content)
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
            except Exception as e:
                if type(e) not in READ_KNOWN_EXCEPTIONS:
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
        # TODO: separate calculation logic from here, to be used in metaheuristics
        stats = self.stats
        print("summary".center(50, "="))
        print("start time:", formatTimeAtShanghai(stats.start_time), sep="\t")
        print("end time:", formatTimeAtShanghai(stats.end_time), sep="\t")
        print("up time:", stats.up_time, sep="\t")
        print("loop count:", stats.loop_count, sep="\t")
        print("total bytes read:", stats.read_bytes, sep="\t")
        print("total bytes write:", stats.write_bytes, sep="\t")
        print("r/w ratio:", stats.rw_ratio)
        print("w/r ratio:", stats.wr_ratio)
        print("read bytes entropy:", stats.read_ent)
        print("write bytes entropy:", stats.write_ent)
        print("r/w entropy ratio:", stats.rw_ent_ratio)
        print("w/r entropy ratio:", stats.wr_ent_ratio)

    def getStatsDict(self):
        start_time = self.start_time
        end_time = time.time()
        up_time = end_time - self.start_time
        read_ent = self.read_entropy_calc.entropy
        write_ent = self.write_entropy_calc.entropy
        loop_count = self.loop_count
        rw_ratio, wr_ratio = leftAndRightSafeDiv(self.read_bytes, self.write_bytes)
        rw_ent_ratio, wr_ent_ratio = leftAndRightSafeDiv(read_ent, write_ent)
        statsDict = dict(
            start_time=start_time,
            end_time=end_time,
            up_time=up_time,
            loop_count=loop_count,
            read_ent=read_ent,
            read_bytes=self.read_bytes,
            write_bytes=self.write_bytes,
            write_ent=write_ent,
            rw_ratio=rw_ratio,
            wr_ratio=wr_ratio,
            rw_ent_ratio=rw_ent_ratio,
            wr_ent_ratio=wr_ent_ratio,
        )
        return statsDict

    @property
    def stats(self):
        # TODO: calculate recent statistics, not just full statistics
        # somehow cached.
        if not (
            isinstance(self._stats, self.actorStatsClass)
            and self._stats.loop_count == self.loop_count
        ):
            statsDict = self.getStatsDict()
            self._stats = self.actorStatsClass(**statsDict)
        return self._stats

    def loop(self):
        self.read()
        self.write(NaiveVocab.generate())
        return True

    def init_check(self):
        """
        Check or wait until the interactive program emits expected output.
        """
        ret = func_timeout.func_timeout(self.max_init_time, self._init_check)
        print("init check passed")
        return ret

    def _init_check(self):
        """
        Implementation of init checks.
        """
        ...

    def heartbeat(self):
        # to prove the program as if still running.
        # do not override this method, unless you know what you are doing.
        access_time = heartbeat_base(uuid = actor_uuid, action = 'heartbeat', pid=current_pid, role='client')
        print('beat at:', access_time)
        return True


    def run(self):
        loop = True
        try:
            self.init_check()
        except:
            print("init check failed")
            log_and_print_unknown_exception()
            raise InteractiveChallengeFailed(
                f"Failed to pass init challenge of: {self.__class__.__name__}"
            )
        while self.heartbeat():
            loop = self.loop()
            if loop is True:
                print(f"[loop\t{str(self.loop_count)}]".center(60, "-"))
                self.loop_count += 1
            else:
                break
        print(
            f"{'heartbeat' if loop else 'loop'} failed.\nexiting at #{self.loop_count}."
        )


def run_naive(cls):
    actor = cls(f"{sys.executable} naive_interactive.py")
    actor.run()


if __name__ == "__main__":
    run_naive(NaiveActor)
