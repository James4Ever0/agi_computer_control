# use websocket to connect to test server.
# currently do not do anything fancy. please!

# from websockets.sync import client

# the language used here is called "godlang", an agent language designed for console, GUI and robot manipulations.

from typing import Optional
import websockets
import asyncio
import json
import beartype

CURSOR = "<|pad|>"

prompt = f"""You are a terminal operator under VT100 environment.

Cursor location will be indicated by {CURSOR}.

Avaliable special codes:

BACKSPACE TAB ENTER ESC PGUP PGDN END HOME LEFT RIGHT INS DEL
CTRL+A ... CTRL+Z
CTRL+0 ... CTRL+9
F1 ... F12

Besides for the built-in special codes, you can also directly write VT100 commands.

Avaliable commands:

TYPE VIEW WAIT

Syntax: 

Each line you generate will be either treated as a single special code or normal string input. The only way to write a newline is to use "ENTER" special code.

If you want to write special code as literal strings, you can use a special command "TYPE", use it like this: `TYPE <special code>`

By default you can only receive the changed lines each turn. If you want to view the whole screen, you can use "VIEW" command. Anything after "VIEW" command will be discarded. Next turn will show you the full screen.

If you do not want to take actions, use "WAIT" command like this: `WAIT <duration in seconds>`

Example 1: Hello world

echo "Hello world!"
ENTER

Example 2: Special prefix usage, will print "ENTER"

echo 
TYPE ENTER
ENTER

Example 3: View the full screen, instead of only showing the changed lines

echo "View the screen"
ENTER
VIEW

Example 4: Wait for 3 seconds

WAIT 3

"""
action_view = False


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


@beartype.beartype
async def recv(ws: websockets.WebSocketClientProtocol):
    global action_view
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
            cursor = data["c"]
            cursor = (cursor[1], cursor[0])
            print("cursur at:", cursor)
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
            print("updated content:")
            print(updated_screen)
            print("updated lines:", *updated_linenos)
            if action_view:
                print("fullscreen:")
                print(dump_full_screen(screen_by_line, cursor))
                action_view = False
            parse_failed = False
        except Exception as e:
            print(e)
        if parse_failed:
            print(data)
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
    "BACKSPACE": BACKSPACE,
    "TAB": TAB,
    "ENTER": "\n",
    "ESC": ESC,
    "PGUP": CSI + "5~",
    "PGDN": CSI + "6~",
    "END": CSI + "4~",
    "HOME": CSI + "1~",
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

COMMANDS = ["TYPE", "VIEW", "WAIT"]


@beartype.beartype
def translate_special_codes(cmd: str):
    return SPECIAL_CODES.get(cmd, cmd)


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
    elif action == "type":
        ret = command_content["data"]
    return ret


async def main(port=8028, regular_sleep_time=1):
    # global action_view
    # command_list = ["i", "Hello world!", "\u001b", ":q!"]
    # command_list = ["i", "Hello world!", "ESC", ":q!"]
    command_list = [
        "echo 'hello world'",
        "ENTER",
        "WAIT 1",
        "TYPE echo 'hello world'",
        "ENTER",
        "VIEW",
        "echo 'hello world'",
        "ENTER",
    ]
    async with websockets.connect(
        f"ws://localhost:{port}/ws"
    ) as ws:  # can also be `async for`, retry on `websockets.ConnectionClosed`
        recv_task = asyncio.create_task(recv(ws))
        for cmd in command_list:
            command_content = handle_command(cmd)
            break_exec = False
            if command_content:
                translated_cmd = await execute_command(command_content)
                if command_content['action'] == 'view':
                    break_exec = True
            else:
                translated_cmd = translate_special_codes(cmd)
            print("Regular sleep for %f seconds" % regular_sleep_time)
            await asyncio.sleep(regular_sleep_time)

            if break_exec:
                print("Exiting reading action list because of 'VIEW' command")
                break
            if translated_cmd:
                await ws.send(translated_cmd)
        await ws.close()
        await recv_task


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
