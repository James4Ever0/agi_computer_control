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


# TODO: submit to tty2img issues
pyte_extended_colormap = {
    "black": "#000000",
    "red": "#cd0000",
    "green": "#00cd00",
    "brown": "#cdcd00",   # Dark yellow (ANSI standard "brown")
    "blue": "#0000cd",
    "magenta": "#cd00cd",
    "cyan": "#00cdcd",
    "white": "#e5e5e5",   # Light gray (not pure white)
    "brightblack": "#7f7f7f",   # Medium gray
    "brightred": "#ff0000",
    "brightgreen": "#00ff00",
    "brightbrown": "#ffff00",   # Bright yellow
    "brightblue": "#0000ff",
    "brightmagenta": "#ff00ff",
    "brightcyan": "#00ffff",
    "brightwhite": "#ffffff"
}
import PIL.ImageColor
PIL.ImageColor.colormap.update(pyte_extended_colormap)


# you have some choices to get a image screenshot of the terminal:
# 1. take a png screenshot from pyte screen buffer, if there is a library for that (https://pypi.org/project/tty2img/)
# 2. use asciinema png/mp4/gif generator like agg etc

# if you stick to text based terminal screen representation, you would either end up with a long, messy, and hard to read text file (char by char, each char with its own style). or a very simple text file that may lose some details such as cursor location, selection, etc.

# if you want to retain the data of the terminal screen, you would need to encode the full data of the entire terminal buffer and reduce its dimensions using a encoder neural network for language model to process.

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
    
    def show_screen_as_image(self):
        # use opencv-python and tty2img to render the screen as an image
        import cv2
        import tty2img
        import numpy

        # if you use tty2img, the color specified in the screen could be missing in pillow
        # the only way to resolve this is to iterate all character object in the screen buffer and replace color name with rgb color code.

        pillow_img = tty2img.tty2img(self.screen, fgDefaultColor="white", showCursor=True) # this requires library "fontconfig" installed on your system
        # the output is either green or black. no colorful output.
        cv2_img = cv2.cvtColor(numpy.array(pillow_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("screen", cv2_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def read_screen(self, insert_cursor=True, cursor_marker="<|cursor|>"):
        """Read the entire screen area and return"""
        screen_cursor = (
            self.screen.cursor
        )  # attrs: [x: int, y: int, attrs: Char, hidden: bool]
        screenshot = ""
        for y in range(self.screen.lines):
            line = self.screen.buffer[y]
            for x in range(self.screen.columns):
                if insert_cursor:
                    if x == screen_cursor.x and y == screen_cursor.y:
                        screenshot += cursor_marker
                char_object = line[
                    x
                ]  # attrs: [data: str, fg: str, bg: str, bold: bool, underline:bool, reverse:bool, blink:bool, invisible:bool, italic:bool, strikethrough:bool]
                screenshot += char_object.data
            # insert a newline
            screenshot += "\n"
        cursor_location = dict(x=screen_cursor.x, y=screen_cursor.y)
        return dict(screenshot=screenshot, cursor_location=cursor_location)


asciinema_record_file = (
    # "../record/terminal/terminal_record_20250807_164804/terminal.cast"
    "./vim_rectangle_selection_test/vim_rectangle_selection.cast" # by rendering this vim recording, we have found that pyte does not work as intended at vim interface. strange characters show up.
    # therefore we decide to stick with asciinema rust libraries {agg, rvt}
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
                    # now, display the terminal screen as image
                    pyte_terminal.show_screen_as_image()
                else:
                    print("PyTE terminal not ready. Unable to insert output event.")
            elif event_type == "i":
                print("Input event found.")
                ...  # handle input event
            input("Press enter to continue...")
        else:
            print("Unknown type %s:" % type(data), data)
