# the executor shall only perform actions
# reference in current project: ./containerized_chatgpt_agent/ptyproc.py

import ptyprocess
import pyte 
import time
import threading

def decode_bytes_to_utf8_safe(_bytes:bytes):
    """
    Decode with UTF-8, but replace errors with a replacement character (�).
    """
    ret = _bytes.decode('utf-8', errors="replace")
    return ret

class PyTEScreen:
    def __init__(self, width, height):
        """
        Initializes the screen with a given width and height.
        Args:
            width: Width of the screen
            height: Height of the screen
        """
        self.screen = pyte.Screen(width, height)
        self.stream = pyte.Stream(self.screen)
    
    def write_bytes(self, _bytes:bytes):
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

class PyTEPTYProcess:
    def __init__(self, command:list[str], width:int, height:int):
        """
        Initializes the terminal emulator with a command to execute.
        Args:
            command: List of command strings to execute in the terminal
        """
        rows, cols = height, width
        self.pty_process = ptyprocess.PtyProcess.spawn(command, dimensions=(rows, cols))
        self.pyte_screen = PyTEScreen(width=width, height=height)
        self.pty_process_reading_thread = threading.Thread(target=self.read_and_update_screen, daemon=True)

    def start_ptyprocess_reading_thread(self):
        """Starts a thread to read output from the terminal process and update the Pyte screen"""
        self.pty_process_reading_thread.start()
        
    def write(self, data: bytes):
        """Writes input data to the terminal process"""
        self.pty_process.write(data)
        
    def read_and_update_screen(self):
        """Reads available output from terminal and updates Pyte screen"""
        try:
            # ptyprocess.read is blocking. only pexpect has read_nonblocking
            process_output_bytes = self.pty_process.read(1024) 
            # write bytes to pyte screen
            self.pyte_screen.write_bytes(process_output_bytes)
        except:
            # Timeout means no data available, EOF means process ended
            pass

class TerminalAsciicastRecordExecutor:
    def __init__(self, command:list[str], width:int, height:int):
        """
        Initializes executor with a command to run in terminal emulator
        Args:
            command: List of command strings to execute
        """
        self.pyte_ptyprocess = PyTEPTYProcess(command=command, width=width, height=height)
    
    def input(self, text: str):
        """Sends input text to the terminal process"""
        self.pyte_ptyprocess.write(text.encode())
        # Allow time for processing output
        time.sleep(0.1)
        # Update screen after input
        self.pyte_ptyprocess.read_and_update_screen()

def test_terminal_executor_using_terminal_recorder_container():
    SLEEP_INTERVAL = 0.1
    CONTAINER_NAME = "terminal_recorder_ttyd"
    ...

def test_harmless_command_locally_with_bash():
    SLEEP_INTERVAL=1
    command = ['bash']
    input_events = ['echo "Hello World!"', "\n"]
    executor = TerminalAsciicastRecordExecutor(command=command, width=80, height=24)
    for event in input_events:
        executor.input(event)
        time.sleep(SLEEP_INTERVAL)
    print("Done")
if __name__ == "__main__":
    # test_terminal_executor_using_terminal_recorder_container()
    test_harmless_command_locally_with_bash()