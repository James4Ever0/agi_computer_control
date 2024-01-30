# use websocket to connect to test server.
# currently do not do anything fancy. please!

# from websockets.sync import client
import websockets
import asyncio
import json
import beartype

@beartype.beartype
async def recv(ws:websockets.WebSocketClientProtocol):
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
            cursor = data['c']
            lines = data['lines'] # somehow it only send updated lines.
            screen = ""
            for lineno, elems in lines:
                for char, _, _, _ in elems:
                    screen += char
                screen += "\n"
            print(screen)
            parse_failed = False
        except:
            pass
        if parse_failed:
            print(data)
            print("!!!!FAILED TO PARSE RESPONSE AS JSON!!!!")


async def main():
    command_list = ["i", "Hello world!", "\u001b", ":q!"]
    async with websockets.connect(
        "ws://localhost:8028/ws"
    ) as ws:  # can also be `async for`, retry on `websockets.ConnectionClosed`
        recv_task = asyncio.create_task(recv(ws))
        for cmd in command_list:
            print("Sending command: " + cmd)
            await ws.send(cmd)
            await asyncio.sleep(1)
        await ws.close()
        await recv_task

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())