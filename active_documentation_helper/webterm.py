"""
    webterm
    ~~~~~~~

    An example showing how to use :mod:`pyte` to implement a basic
    single-user web terminal.

    Client-side ``webterm.js`` supports
    * incremental rendering via :data:`~pyte.screens.DiffScreen.dirty`,
    * most of the common keyboard events,
    * pagination on Meta + P/Meta + A.

    .. note:: This example requires at least Python 3.7 and version 3.0 of the
              ``aiohttp`` library.

    .. seealso::

       `The TTY demystified <http://www.linusakesson.net/programming/tty>`_
       for an introduction to the inner workings of the TTY subsystem.

    :copyright: (c) 2017 by pyte authors and contributors,
                see AUTHORS for details.
    :license: LGPL, see LICENSE for more details.
"""

import json
import os
import pty
import shlex
import signal
import webbrowser
from pathlib import Path
import uuid
import pydantic

import aiohttp
import asyncio
from aiohttp import web

import pyte
import functools

# TODO: only send actions from client side, do not send message, to reduce complexity


class TerminalClientEvent(pydantic.BaseModel):
    action: str
    message: str
    timestamp: int
    """
    Unit: miliseconds
    """


class Terminal:
    def __init__(self, columns, lines, p_in):
        self.screen = pyte.HistoryScreen(columns, lines)
        self.screen.set_mode(pyte.modes.LNM)
        self.screen.write_process_input = lambda data: p_in.write(data.encode())
        self.stream = pyte.ByteStream()
        self.stream.attach(self.screen)

    def feed(self, data):
        self.stream.feed(data)

    def dumps(self):
        cursor = self.screen.cursor
        lines = []
        for y in self.screen.dirty:
            line = self.screen.buffer[y]
            data = [
                (char.data, char.reverse, char.fg, char.bg)
                for char in (line[x] for x in range(self.screen.columns))
            ]
            lines.append((y, data))

        self.screen.dirty.clear()
        return json.dumps(
            {"type": "update", "data": {"c": (cursor.x, cursor.y), "lines": lines}}
        )
        # return json.dumps({"c": (cursor.x, cursor.y), "lines": lines})


# def open_terminal(command="bash", columns=80, lines=24):
def open_terminal(command="vim", columns=80, lines=24):
    p_pid, master_fd = pty.fork()
    if p_pid == 0:  # Child.
        argv = shlex.split(command)
        env = dict(
            TERM="linux", LC_ALL="en_GB.UTF-8", COLUMNS=str(columns), LINES=str(lines)
        )
        os.execvpe(argv[0], argv, env)

    # File-like object for I/O with the child process aka command.
    p_out = os.fdopen(master_fd, "w+b", 0)
    return Terminal(columns, lines, p_out), p_pid, p_out


def generate_terminal_identifier():
    ret = str(uuid.uuid4())
    return ret


# TODO: link log, print data & exception along with terminal_identifier
# TODO: send the client terminal identifier and show as webpage title


async def websocket_handler(request, command: str, view_interval: int = 2000):
    """
    view_interval: miliseconds
    """
    ws = web.WebSocketResponse()
    datalist: list[TerminalClientEvent] = []
    await ws.prepare(request)

    terminal_identifier = generate_terminal_identifier()

    request.app["websockets"].add(asyncio.current_task())

    terminal, p_pid, p_out = open_terminal(command)
    await ws.send_str(json.dumps({"type": "identifier", "data": terminal_identifier}))
    await ws.send_str(terminal.dumps())

    def on_master_output():
        success = False
        try:
            out_content = p_out.read(65536)
            terminal.feed(
                out_content
            )  # should you send message to the client, and end this session, or just close this websocket.
            asyncio.create_task(ws.send_str(terminal.dumps()))
            success = True
        except IOError:
            print(
                f"Closing terminal <{terminal_identifier}> because unable to read from process output."
            )
        finally:
            if not success:
                print(f"Closing websocket <{terminal_identifier}> unexpectedly.")
                asyncio.create_task(ws.close())

    loop = asyncio.get_event_loop()
    loop.add_reader(p_out, on_master_output)
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                if msg.data == pyte.control.ESC + "N":
                    terminal.screen.next_page()
                    ws.send_str(terminal.dumps())
                elif msg.data == pyte.control.ESC + "P":
                    terminal.screen.prev_page()
                    ws.send_str(terminal.dumps())
                else:
                    p_out.write(msg.data.encode())
            elif msg.type == aiohttp.WSMsgType.BINARY:
                # try parsing as JSON
                try:
                    data = TerminalClientEvent.parse_raw(msg.data)
                    print(f"Client <{terminal_identifier}> event:", data)
                    datalist.append(data)
                except:
                    print(
                        f"Unable to parse client <{terminal_identifier}> event:",
                        msg.data,
                    )
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise ws.exception()
    except (asyncio.CancelledError, OSError):  # Process died?
        pass
    finally:
        # convert datalist into godscript
        # TODO: record terminal screen, interject into the godscript
        # TODO: differentiate agent actions from terminal observations
        # TODO: figure out how to handle the WAIT command and the time alignment
        godscript = []
        if len(datalist) > 0: # you should put more things than just agent actions into this list, but also environment actions (feedback)
            init_time = datalist[0].timestamp
            last_time = init_time
            for data in datalist:
                current_time = data.timestamp
                wait_time = current_time - last_time
                if wait_time != 0:
                    godscript.append(f"WAIT {wait_time/1000}")
                last_time = current_time
                if (current_time - init_time) > view_interval:
                    godscript.append("VIEW")
                    init_time = current_time
                if data.action == "TYPE":
                    godcommand = f"TYPE {data.message}"
                    # can we use special token for space?
                else:  # special
                    godcommand = data.action
                godscript.append(godcommand)
            if godscript[-1] != "VIEW":
                godscript.append("VIEW")
        print("GODSCRIPT DUMP".center(60, "="))
        print("\n".join(godscript))
        loop.remove_reader(p_out)
        os.kill(p_pid, signal.SIGTERM)
        p_out.close()
        if not is_shutting_down:
            request.app["websockets"].remove(asyncio.current_task())
    print(f"Client <{terminal_identifier}> exiting.")
    await ws.close()
    return ws


is_shutting_down = False


async def on_shutdown(app):
    """Closes all WS connections on shutdown."""
    global is_shutting_down
    is_shutting_down = True
    for task in app["websockets"]:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    # parameters: port, command, launch browser or not (headless)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8079)
    parser.add_argument("-c", "--command", type=str, default="vim")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    port = args.port
    command = args.command
    print("Shell command: " + command)
    headless = args.headless
    app = web.Application()
    app["websockets"] = set()
    app.router.add_get("/ws", functools.partial(websocket_handler, command=command))
    app.on_shutdown.append(on_shutdown)
    if not headless:
        app.router.add_static("/", Path(__file__).parent / "static", show_index=True)
        webbrowser.open_new_tab(f"http://localhost:{port}/index.html")
    web.run_app(app, port=port)
