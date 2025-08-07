# asciinema seems to provide a client-side javascript player
# https://docs.asciinema.org/manual/player/quick-start/
# you may want to know how the data is read from asciicast format from the player library.
# npm install --save-dev asciinema-player@3.6.3

# we may want to load the player to jsdom. otherwise electron is needed.

# in v3 format two new events are introduced: resize (r) and exitcode (x)
# https://docs.asciinema.org/manual/asciicast/v3/

# but as a record viewer for ai, we need something like screenshots, cursor location, selections, etc

# prioritize the easiest one to implement, by just rendering the text out.
# or use existing pyte terminal renderer.

# find usage of pyte elsewhere in this project (global search: "import pyte")

import pyte
import os
import json


class PyTETerminal:
    def __init__(self, rows, cols):
        self.rows = rows
        self.lines = rows
        self.cols = cols
        self.columns = cols
        self.size = "%sx%s" % (cols, rows)
        self.screen = pyte.Screen(lines=rows, columns=cols)
        self.stream = pyte.Stream(self.screen)

    def write(self, payload: str):
        """Write the payload (str) to the screen"""
        self.stream.feed(payload)

    def read_screen(self, insert_cursor = True, cursor_marker ="<|cursor|>"):
        """Read the entire screen area and return"""
        screen_cursor = self.screen.cursor
        screenshot = ""
        for y in range(self.screen.lines):
            line = self.screen.buffer[y]
            for x in range(self.screen.columns):
                if insert_cursor:
                    if x == screen_cursor.x and y == screen_cursor.y:
                        screenshot += cursor_marker
                char_object = line[x]
                screenshot += char_object.data
            # insert a newline
            screenshot += "\n"
        cursor_location = dict(x=screen_cursor.x, y=screen_cursor.y)
        return dict(screenshot=screenshot, cursor_location=cursor_location)


asciinema_record_file = (
    "../record/terminal/terminal_record_20250807_164804/terminal.cast"
)

assert os.path.isfile(asciinema_record_file)

print("Parsing ASCIICAST v2 record:", asciinema_record_file)
pyte_terminal = None
# now proceed to parsing, line by line

with open(asciinema_record_file, "r") as f:
    for line in f.readlines():
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print("Failed to decode line as JSON:")
            print("", line)
            continue
        if type(data) == dict:
            # metadata
            print("Metadata:", data)
            # create a pyte terminal now.
            if not pyte_terminal:
                cols = data["width"]
                rows = data["height"]
                print(
                    "Initializing PyTE Terminal with cols: %s, rows: %s" % (cols, rows)
                )
                pyte_terminal = PyTETerminal(cols=cols, rows=rows)
            else:
                print("PyTE terminal is already initialized")
        elif type(data) == list:
            # event stream
            print("Event:", data)
            event_type = data[1]
            event_payload = data[2]  # str
            if event_type == "o":
                print("Output event found.")
                if pyte_terminal:
                    # insert output into pyte terminal
                    pyte_terminal.write(event_payload)
                    screen_data = pyte_terminal_screenshot = pyte_terminal.read_screen()
                    # clear the screen, print the screenshot and cursor position
                    os.system("clear")
                    print("Cursor location: %s" % screen_data["cursor_location"])
                    print("Screenshot (size: %s)" % pyte_terminal.size)
                    print(screen_data["screenshot"])
                else:
                    print("PyTE terminal not ready. Unable to insert output event.")
            elif event_type == "i":
                print("Input event found.")
                ...# handle input event
            input("Press enter to continue...")
        else:
            print("Unknown type %s:" % type(data), data)
