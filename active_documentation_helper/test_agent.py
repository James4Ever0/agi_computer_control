# use websocket to connect to test server.
# currently do not do anything fancy. please!

# from websockets.sync import client

# the language used here is called "godlang", an agent language designed for console, GUI and robot manipulations.

from typing import Callable, Optional
from numbers import Number
import websockets
import asyncio
import json
import beartype
import llm

CURSOR = "<|pad|>"

# TODO: agent says it wants an new command "CURSOR <x> <y>"
# TODO: collect the data or chaos the agent has made along the way, for better investigation, or new features it wants.
# TODO: collect human operational data
# TODO: collect random data based on syntax
# TODO: verify it is possible for human to operate and achieve targets in such environment, then later we would record our successful attempts to let the agent learn later.
# TODO: send agent action to the server
# TODO: specify view line range instead of viewing the full screen
# TODO: specify more types of view commands

INIT_PROMPT = f"""You are a terminal operator under VT100 environment.

The console is running under Alpine Linux. You can access busybox utils.

Cursor location will be indicated by {CURSOR}. Do not write {CURSOR} by yourself.

Avaliable special codes: (do not prefix these codes with 'TYPE' when you want to use them)

BACKSPACE SPACE TAB ENTER ESC PGUP PGDN END HOME LEFT RIGHT UP META+UP DOWN META+DOWN INS DEL
CTRL+A ... CTRL+Z
CTRL+0 ... CTRL+9
F1 ... F12

Besides for the built-in special codes, you can also directly write VT100 commands, without using 'echo'.

Avaliable commands:

TYPE VIEW WAIT REM SPECIAL CURSOR

Syntax: 

Each line you generate will be either treated as a single action or normal chat text. The only way to write a newline is to use "ENTER" special code, with the special prefix "SPECIAL".

You use "SPECIAL <special_code>" to use special codes.

You can use a special command "TYPE", to type string into terminal. Use it like this: `TYPE <string>`. The string can be normal text, including special code.

By typing CURSOR <x> <y> you can move the cursor around the terminal. Remeber to put the cursor with in the terminal, otherwise it would not be a valid position.

If you want to move the cursor relatively better use special keys and hot keys.

By default you can only receive the changed lines each turn. If you want to view the whole screen, you can use "VIEW" command. Anything after "VIEW" command will be discarded. Next turn will show you the full screen.

If you want to wait for some time, use "WAIT" command like this: `WAIT <duration in seconds>`

If you do not want to take actions, use "REM" command like: `REM <comments>`

Example 1: Hello world

TYPE echo "Hello world!"
SPECIAL ENTER

Example 2: Special prefix usage, will print "ENTER"

TYPE echo 
TYPE ENTER
SPECIAL ENTER

Example 3: View the full screen, instead of only showing the changed lines

TYPE echo "View the screen"
SPECIAL ENTER
VIEW

Example 4: Wait for 3 seconds

WAIT 3

Example 5: Use comment

REM I will sleep for a second, then view the full screen
WAIT 1
VIEW

"""
action_view = False
import time


def get_timestamp():
    return round(time.time() * 1000)


@beartype.beartype
def insert_cursor_at_column(line: str, column: int):
    line_with_cursor = list(line)
    line_with_cursor.insert(column, CURSOR)
    return "".join(line_with_cursor)


@beartype.beartype
def dump_full_screen(screen_by_line: dict[int, str], cursor: Optional[tuple[int, int]]):
    screen = ""
    for k in sorted(screen_by_line.keys()):
        line = screen_by_line[k]
        if cursor is not None:
            if cursor[0] == k:
                line = insert_cursor_at_column(line, cursor[1])
        screen += line
        screen += "\n"
    return screen


observations = []


def build_prompt():
    global observations
    last_updated_observation = None
    last_full_screen_observation = None

    for it in reversed(observations):
        if it["type"] == "update":
            if last_updated_observation:
                last_updated_observation = it["data"]

        if it["type"] == "full_screen":
            if last_full_screen_observation:
                last_full_screen_observation = it["data"]

    # use last updated observation & last full screen observation
    components = []
    if last_updated_observation:
        comp = f"""Updated lines: (format: [lineno] <content>)

{last_updated_observation}
"""
        components.append(comp)
    if last_full_screen_observation:
        comp = f"""Full screen:

{last_full_screen_observation}
"""
        components.append(comp)
    comp = f"""Your input to the terminal, according to syntax given above: (always prefix your talk with 'REM', and return 1 to 2 commands each turn, in order to get instant feedback from terminal)
"""
    components.append(comp)
    prompt = "\n".join(components)
    observations.clear()
    return prompt


@beartype.beartype
async def recv(ws: websockets.WebSocketClientProtocol):
    global action_view, observations
    screen_by_line = {}
    while not ws.closed:
        # Background task: Infinite loop for receiving data
        try:
            data = await ws.recv()  # Replace with your actual receive function
        except websockets.exceptions.ConnectionClosedOK:
            print("Connection closed. Exiting.")
            break
        print("====================JSON RESPONSES====================")
        parse_failed = True
        try:
            data = json.loads(data)
            data_type = data.get("type", "unknown")
            if data_type == "unknown":
                raise Exception(
                    "unknown data received  from websocket", str(data)[:60] + " ..."
                )
            elif data_type == "identifier":
                print("identifier received from websocket", data["data"])
            else:
                data = data["data"]
                cursor = data["c"]
                cursor = (cursor[1], cursor[0])
                print("Cursur at:", cursor)
                lines = data["lines"]  # somehow it only send updated lines.
                updated_screen = ""
                updated_linenos = []
                for lineno, elems in lines:
                    line = ""
                    updated_linenos.append(lineno)
                    for char, _, _, _ in elems:
                        line += char
                    screen_by_line[lineno] = line
                    updated_screen_line = line

                    if lineno == cursor[0]:
                        updated_screen_line = insert_cursor_at_column(
                            updated_screen_line, cursor[1]
                        )
                    updated_screen += f"[{str(lineno).center(2)}] {updated_screen_line}"
                    updated_screen += "\n"
                print("Updated content:")
                print(updated_screen)
                observations.append({"type": "update", "data": updated_screen})
                print("Updated lines:", *updated_linenos)
                print("Fullscreen:")
                fullscreen = dump_full_screen(screen_by_line, cursor)
                print(fullscreen)
                if action_view:
                    observations.append({"type": "full_screen", "data": fullscreen})
                    action_view = False
            parse_failed = False
        except Exception as e:
            raise e
        if parse_failed:
            print(str(data)[:60])
            print("!!!!FAILED TO PARSE RESPONSE AS JSON!!!!")


BACKSPACE = "\u0008"
TAB = "\u0009"
ESC = "\u001b"
CSI = ESC + "["
F_N = {
    1: CSI + "[A",
    2: CSI + "[B",
    3: CSI + "[C",
    4: CSI + "[D",
    5: CSI + "[E",
    6: CSI + "17~",
    7: CSI + "18~",
    8: CSI + "19~",
    9: CSI + "20~",
    10: CSI + "21~",
    11: CSI + "23~",
    12: CSI + "24~",
}
CONTROL_N = {
    0: "\u0030",
    1: "\u0031",
    2: "\u0000",
    3: "\u001b",
    4: "\u001c",
    5: "\u001d",
    6: "\u001e",
    7: "\u001f",
    8: "\u007f",
    9: "\u0039",
}

SPECIAL_CODES = {
    "SPACE": " ",
    "BACKSPACE": BACKSPACE,
    "TAB": TAB,
    "ENTER": "\n",
    "ESC": ESC,
    "PGUP": CSI + "5~",
    "PGDN": CSI + "6~",
    "END": CSI + "4~",
    "HOME": CSI + "1~",
    "UP": CSI + "A",
    "META+UP": ESC + "P",
    "DOWN": CSI + "B",
    "META+DOWN": ESC + "N",
    "LEFT": CSI + "D",
    "RIGHT": CSI + "C",
    "INS": CSI + "2~",
    "DEL": CSI + "3~",
}

FN_CODES = {
    "F1": F_N[1],
    "F2": F_N[2],
    "F3": F_N[3],
    "F4": F_N[4],
    "F5": F_N[5],
    "F6": F_N[6],
    "F7": F_N[7],
    "F8": F_N[8],
    "F9": F_N[9],
    "F10": F_N[10],
    "F11": F_N[11],
    "F12": F_N[12],
}

CTRL_CODES = {
    "CTRL+A": chr(ord("A") - 65 + 1),
    "CTRL+B": chr(ord("B") - 65 + 1),
    "CTRL+C": chr(ord("C") - 65 + 1),
    "CTRL+D": chr(ord("D") - 65 + 1),
    "CTRL+E": chr(ord("E") - 65 + 1),
    "CTRL+F": chr(ord("F") - 65 + 1),
    "CTRL+G": chr(ord("G") - 65 + 1),
    "CTRL+H": chr(ord("H") - 65 + 1),
    "CTRL+I": chr(ord("I") - 65 + 1),
    "CTRL+J": chr(ord("J") - 65 + 1),
    "CTRL+K": chr(ord("K") - 65 + 1),
    "CTRL+L": chr(ord("L") - 65 + 1),
    "CTRL+M": chr(ord("M") - 65 + 1),
    "CTRL+N": chr(ord("N") - 65 + 1),
    "CTRL+O": chr(ord("O") - 65 + 1),
    "CTRL+P": chr(ord("P") - 65 + 1),
    "CTRL+Q": chr(ord("Q") - 65 + 1),
    "CTRL+R": chr(ord("R") - 65 + 1),
    "CTRL+S": chr(ord("S") - 65 + 1),
    "CTRL+T": chr(ord("T") - 65 + 1),
    "CTRL+U": chr(ord("U") - 65 + 1),
    "CTRL+V": chr(ord("V") - 65 + 1),
    "CTRL+W": chr(ord("W") - 65 + 1),
    "CTRL+X": chr(ord("X") - 65 + 1),
    "CTRL+Y": chr(ord("Y") - 65 + 1),
    "CTRL+Z": chr(ord("Z") - 65 + 1),
    "CTRL+0": CONTROL_N[0],
    "CTRL+1": CONTROL_N[1],
    "CTRL+2": CONTROL_N[2],
    "CTRL+3": CONTROL_N[3],
    "CTRL+4": CONTROL_N[4],
    "CTRL+5": CONTROL_N[5],
    "CTRL+6": CONTROL_N[6],
    "CTRL+7": CONTROL_N[7],
    "CTRL+8": CONTROL_N[8],
    "CTRL+9": CONTROL_N[9],
}

SPECIAL_CODES.update(FN_CODES)
SPECIAL_CODES.update(CTRL_CODES)

COMMANDS = ["TYPE", "VIEW", "WAIT", "REM", "SPECIAL", "CURSOR"]
# breakpoint()


@beartype.beartype
def translate_special_codes(cmd: str):
    prefix = "SPECIAL "
    if cmd.startswith(prefix):
        special_code = cmd[len(prefix) :].strip()
        return SPECIAL_CODES.get(special_code, None)  # do not handle non-special code


def handle_command(cmd: str):
    command_content = {}
    if cmd == "VIEW":
        command_content["action"] = "view"
    elif cmd.startswith("TYPE "):
        command_content["action"] = "type"
        command_content["data"] = cmd[5:]
    elif cmd.startswith("WAIT "):
        data = cmd[5:].strip()
        try:
            data = float(data)
            command_content["action"] = "wait"
            command_content["data"] = data
        except:
            pass
    elif cmd.startswith("REM "):
        data = cmd[4:].strip()
        command_content["action"] = "rem"
        command_content["data"] = data
    elif cmd.startswith("CURSOR "):
        data = cmd[7:].strip()
        command_content["action"] = "cursor"
        command_content["data"] = data
    # did not handle special?
    return command_content


@beartype.beartype
async def execute_command(command_content: dict):
    global action_view
    action = command_content["action"]
    ret = ""
    if action == "view":
        action_view = True
    elif action == "wait":
        data = command_content["data"]
        print("Waiting for %f seconds" % data)
        await asyncio.sleep(data)
    elif action == "rem":
        data = command_content["data"]
        print("Reminder:", data)
    elif action == "type":
        ret = command_content["data"]
    elif action == "cursor":
        ret = command_content["data"]
        ret = ret.strip().split()
        elems = []
        for elem in ret:
            try:
                elem = int(elem)
                elems.append(elem)
            except:
                pass
        ret = elems
    return ret


@beartype.beartype
def get_command_list(response: str) -> list[str]:
    return response.split("\n")


@beartype.beartype
async def execute_command_list(
    command_list: list[str],
    ws: websockets.WebSocketClientProtocol,
    regular_sleep_time: Number,
):
    for cmd in command_list:
        command_content = handle_command(cmd)
        break_exec = False
        if command_content:
            translated_cmd = await execute_command(command_content)
            if command_content["action"] == "view":
                break_exec = True
            elif command_content["action"] == "type":
                await ws.send(
                    json.dumps(
                        dict(
                            action="TYPE",
                            timestamp=get_timestamp(),
                            message=command_content["data"],
                        )
                    ).encode()
                )
        else:
            translated_cmd = translate_special_codes(cmd)  # special code for sure.
            if translated_cmd:
                await ws.send(
                    json.dumps(
                        dict(action=cmd, timestamp=get_timestamp(), message="")
                    ).encode()
                )
            else:
                print("Chat:", cmd)
        print("Regular sleep for %f seconds" % regular_sleep_time)
        await asyncio.sleep(regular_sleep_time) # type: ignore
        if break_exec:
            print("Exiting reading action list because of 'VIEW' command")
            break
        if translated_cmd:
            if type(translated_cmd) == list:
                # send as cursor move.
                # TerminalClientAction
                await ws.send(
                    json.dumps(dict(name="CURSOR", args=translated_cmd[:2])).encode()
                )
            elif type(translated_cmd) == str:
                await ws.send(translated_cmd)  # executing command
            else:
                raise Exception(
                    f"Unknown translated cmd type: <{type(translated_cmd)}>"
                )


@beartype.beartype
async def main_template(
    command_list_generator: Callable,
    port: int = 8028,
    regular_sleep_time: Number = 1,
    init_sleep_time: Number = 1,
    total_batches: int = 10,
):
    async with websockets.connect(
        f"ws://localhost:{port}/ws"
    ) as ws:  # can also be `async for`, retry on `websockets.ConnectionClosed`
        recv_task = asyncio.create_task(recv(ws))
        await asyncio.sleep(init_sleep_time)
        try:
            for _ in range(
                total_batches
            ):  # do not try infinite loop. please. the webterm is logging data.
                # while True:
                command_list = command_list_generator()
                print("Command list:", command_list)
                await execute_command_list(command_list, ws, regular_sleep_time)
        except KeyboardInterrupt:
            print("Interrupted shell connection")
        finally:
            await ws.close()
            await recv_task


@beartype.beartype
async def main():
    # global action_view
    # command_list = ["i", "Hello world!", "\u001b", ":q!"]
    # command_list = ["i", "Hello world!", "ESC", ":q!"]
    # command_list = [
    #     "echo 'hello world'",
    #     "ENTER",
    #     "WAIT 1",
    #     "TYPE echo 'hello world'",
    #     "ENTER",
    #     "VIEW",
    #     "echo 'hello world'",
    #     "ENTER",
    # ]
    model = llm.LLM(prompt=INIT_PROMPT)

    def command_list_generator():
        query = build_prompt()
        response = model.run_once(query)
        command_list = get_command_list(response)
        return command_list

    await main_template(command_list_generator)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
