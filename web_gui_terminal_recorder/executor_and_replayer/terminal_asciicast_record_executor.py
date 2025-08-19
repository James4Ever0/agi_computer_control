# the executor shall only perform actions
# reference in current project: ./containerized_chatgpt_agent/ptyproc.py

# pyte is deprecated. use asciinema agg or rvt instead.
# links:
# https://github.com/asciinema/agg
# https://github.com/asciinema/avt

# check out https://pypi.org/project/pyxtermjs/ for steps on running pty and connect to xterm.js

# TODO: create a python package for terminal/gui executor, and upload to pypi

import ptyprocess
import pyte
import time
import threading
import agg_python_bindings


# TODO: submit to tty2img issues
pyte_extended_colormap = {
    "black": "#000000",
    "red": "#cd0000",
    "green": "#00cd00",
    "brown": "#cdcd00",  # Dark yellow (ANSI standard "brown")
    "blue": "#0000cd",
    "magenta": "#cd00cd",
    "cyan": "#00cdcd",
    "white": "#e5e5e5",  # Light gray (not pure white)
    "brightblack": "#7f7f7f",  # Medium gray
    "brightred": "#ff0000",
    "brightgreen": "#00ff00",
    "brightbrown": "#ffff00",  # Bright yellow
    "brightblue": "#0000ff",
    "brightmagenta": "#ff00ff",
    "brightcyan": "#00ffff",
    "brightwhite": "#ffffff",
}
import PIL.ImageColor

PIL.ImageColor.colormap.update(pyte_extended_colormap)


def decode_bytes_to_utf8_safe(_bytes: bytes):
    """
    Decode with UTF-8, but replace errors with a replacement character (�).
    """
    ret = _bytes.decode("utf-8", errors="replace")
    return ret


# screen init params: width, height
# screen traits: write_bytes, display, screenshot


class AvtScreen:
    def __init__(self, width: int, height: int):
        self.vt = agg_python_bindings.TerminalEmulator(width, height)

    def write_bytes(self, _bytes: bytes):
        decoded_bytes = decode_bytes_to_utf8_safe(_bytes)
        self.vt.feed_str(decoded_bytes)

    @property
    def cursor(self):
        col, row, visible = self.vt.get_cursor()
        ret = pyte.screens.Cursor(x=col, y=row)
        ret.hidden = not visible
        return ret

    @property
    def display(self):
        ret = "\n".join(self.vt.text_raw())
        return ret

    def screenshot(self, png_output_path: str):
        self.vt.screenshot(png_output_path)


class PyTEScreen:
    def __init__(self, width:int, height:int):
        """
        Initializes the screen with a given width and height.
        Args:
            width: Width of the screen
            height: Height of the screen
        """
        self.screen = pyte.Screen(width, height)
        self.stream = pyte.Stream(self.screen)

    @property
    def cursor(self):
        return self.screen.cursor

    def write_bytes(self, _bytes: bytes):
        """
        Writes bytes to the screen.
        Args:
            bytes: Bytes to write to the screen
        """
        decoded_bytes = decode_bytes_to_utf8_safe(_bytes)
        self.stream.feed(decoded_bytes)

    @property
    def display(self):
        """
        Returns the current state of the screen as a string.
        """
        ret = "\n".join(self.screen.display)
        return ret

    def screenshot(self, png_output_path: str):
        """
        Saves the current state of the screen as a PNG image.
        Args:
            png_output_path: Path to save the PNG image
        """
        # checkout pyte terminal record viewer in this project
        import tty2img
        import PIL.Image

        pillow_img: PIL.Image.Image = tty2img.tty2img(
            self.screen, fgDefaultColor="white", showCursor=True
        )
        pillow_img.save(png_output_path)


class TerminalProcess:
    def __init__(self, command: list[str], width: int, height: int, backend="avt"):
        """
        Initializes the terminal emulator with a command to execute.
        Args:
            command: List of command strings to execute in the terminal
        """
        rows, cols = height, width
        self.pty_process = ptyprocess.PtyProcess.spawn(command, dimensions=(rows, cols))
        if backend == "avt":
            self.vt_screen = AvtScreen(width=width, height=height)
        elif backend == "pyte":
            print("Warning: using deprecated backend '%s'" % pyte)
            self.vt_screen = PyTEScreen(width=width, height=height)
        else:
            raise ValueError(
                "Unknown terminal emulator backend '%s' (known ones: avt, pyte)"
                % backend
            )
        self.__pty_process_reading_thread = threading.Thread(
            target=self.__read_and_update_screen, daemon=True
        )
        self.__start_ptyprocess_reading_thread()

    def __start_ptyprocess_reading_thread(self):
        """Starts a thread to read output from the terminal process and update the Pyte screen"""
        self.__pty_process_reading_thread.start()

    def write(self, data: bytes):
        """Writes input data to the terminal process"""
        self.pty_process.write(data)

    def __read_and_update_screen(self, poll_interval=0.01):
        """Reads available output from terminal and updates Pyte screen"""
        while True:
            try:
                # ptyprocess.read is blocking. only pexpect has read_nonblocking
                process_output_bytes = self.pty_process.read(1024)
                # write bytes to pyte screen
                self.vt_screen.write_bytes(process_output_bytes)
            except KeyboardInterrupt: # user interrupted
                break
            except SystemExit: # python process exit
                break
            except SystemError: # python error
                break
            except EOFError: # terminal died
                break
            except:
                # Timeout means no data available, EOF means process ended
                pass
            finally:
                time.sleep(poll_interval)


class TerminalExecutor:
    def __init__(self, command: list[str], width: int, height: int):
        """
        Initializes executor with a command to run in terminal emulator, using avt as backend.

        Args:
            command: List of command strings to execute
        """
        self.terminal = TerminalProcess(
            command=command, width=width, height=height
        )

    def input(self, text: str):
        """
        Sends input text to the terminal process
        """
        self.terminal.write(text.encode())
        # Allow time for processing output
        time.sleep(0.1)

    @property
    def display(self) -> str:
        return self.terminal.vt_screen.display

    def screenshot(self, png_save_path: str):
        self.terminal.vt_screen.screenshot(png_save_path)


def test_terminal_executor_using_terminal_recorder_container():
    SLEEP_INTERVAL = 0.2
    CONTAINER_NAME = "terminal_recorder_ttyd"  # TODO: start one with custom container name, instead of clashing the cybergod terminal recorder container
    command = ["docker", "exec", "-it", CONTAINER_NAME, "bash"]
    executor = TerminalExecutor(command, width=80, height=24)


def test_harmless_command_locally_with_bash():
    SLEEP_INTERVAL = 0.5
    command = ["docker", "run" , "--rm", "-it", "alpine"]
    input_events = ['echo "Hello World!"', "\n"]
    executor = TerminalExecutor(command=command, width=80, height=24)
    time.sleep(1)
    for event in input_events:
        executor.input(event)
        time.sleep(SLEEP_INTERVAL)
    # check for screenshot, text dump
    text_dump = executor.display
    print("Dumping terminal display to terminal_executor_text_dump.txt")
    with open("terminal_executor_text_dump.txt", "w+") as f:
        f.write(text_dump)
    print("Taking terminal screenshot at terminal_executor_screenshot.png")
    executor.screenshot("terminal_executor_screenshot.png")
    print("Done")

def test():
    # test_terminal_executor_using_terminal_recorder_container()
    test_harmless_command_locally_with_bash()

if __name__ == "__main__":
    test()