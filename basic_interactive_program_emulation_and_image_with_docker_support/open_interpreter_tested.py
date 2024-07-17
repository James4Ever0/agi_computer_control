# patching input function
import typing
import traceback


def custom_input(banner: str, ans: str = "y"):
    print("[*] Query:", banner)
    print("[*] Answer:", ans)

    input("Continue?")

    return ans

import subprocess
import copy

PRINT_PARSING_DETAIL = False
PRINT_MESSAGE_CHUNK = True

SUBPROCESS_TIMEOUT = 20

old_popen = copy.copy(subprocess.Popen)
import threading

class new_popen(old_popen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # type: ignore
        # threading.Thread(target=self.self_destruct, daemon=True).start()
    
    def self_destruct(self):
        print("[*] Initiating self destruct countdown.")
        time.sleep(SUBPROCESS_TIMEOUT)
        if self.returncode is None:
            print(f"[-] Process timeout after {SUBPROCESS_TIMEOUT} seconds.")
            print("[-] Killing process")
            self.kill()
        else:
            print(f"[*] Process exited with code", self.returncode)

setattr(subprocess, "Popen", new_popen)

from interpreter.core.computer.terminal.languages.subprocess_language import SubprocessLanguage
import time

def subprocess_input(self:SubprocessLanguage, command:str, suffix:str= "\n"):
    print("[*] Writing to stdin:", command)
    self.process.stdin.write(command+suffix) # type: ignore
    self.process.stdin.flush() # type: ignore

def yes(self:SubprocessLanguage, interval = 2):
    while True:
        # subprocess_input(self,"y")
        # subprocess_input(self,"")
        time.sleep(interval)

setattr(SubprocessLanguage, "input", subprocess_input)
old_start_process = copy.copy(SubprocessLanguage.start_process)

def new_start_process(self:SubprocessLanguage):
    old_start_process(self)
    print("[*] Starting yes input thread.")
    threading.Thread(target=yes, daemon=True, args=(self, )).start()

setattr(SubprocessLanguage, "start_process", new_start_process)

from interpreter.terminal_interface import terminal_interface

setattr(terminal_interface, "input", custom_input)

from interpreter import OpenInterpreter
from interpreter.core.computer.computer import Computer

interpreter = OpenInterpreter(disable_telemetry = True)


# old_run = copy.copy(computer.run)
class CustomComputer(Computer):
    def run(self, *args, **kwargs):
        print(f"[*] Calling custom computer run method, args={args}, kwargs={kwargs}")
        language, code = args
        ret = super().run(*args, **kwargs) # <generator object Terminal._streaming_run>
        print(f"[*] computer execution result:", ret)
        return ret

computer = CustomComputer(interpreter)
# setattr(computer, 'run', new_run)
interpreter.computer =computer


# from interpreter.core.respond import respond
import pydantic


class MessageContent(pydantic.BaseModel):
    type: typing.Literal["code", "message", "confirmation", "console", "execution"]
    format: typing.Optional[str] = None  # execution, python, bash
    content: typing.Optional[typing.Union[int, str, "MessageContent"]] = None
    start: typing.Optional[bool] = None
    end: typing.Optional[bool] = None


class MessageChunk(MessageContent):
    role: typing.Literal["assistant", "terminal", "computer"]


# it works by manual tweaking and prompt engineering
# like a rule engine with slightest intelligence

import os

# ref: https://docs.openinterpreter.com/getting-started/introduction

MODEL = "openai/mixtral-local"
TEMPERATURE = 0.1
API_BASE = os.environ.get("OPENAI_API_BASE", "http://localhost:8101/v1")

# prompt = "get me the ip address of bing.com. all programs can run with root permission without sudo."

prompt = "install nmap command. all programs always run with root permission without sudo prefix."

interpreter.offline = True
interpreter.llm.temperature = TEMPERATURE  # type:ignore
interpreter.llm.model = MODEL
interpreter.llm.api_base = API_BASE
interpreter.llm.api_key = os.environ.get("OPENAI_API_KEY", "dummy_key")

# auto code execution
# intercept "input" with hooks
# interpreter.auto_run = True


# it is not programmatically interactive
# find that 'Would you like to' thing.
# ref: /home/pi/mambaforge/envs/cybergod/lib/python3.10/site-packages/interpreter/terminal_interface/terminal_interface.py:262

# code execution happens with "computer" role, "console" type, using "interpreter.computer.run"
# ref: /home/pi/mambaforge/envs/cybergod/lib/python3.10/site-packages/interpreter/core/respond.py:237

# the terminal is real time streaming as well.
interpreter.responding = True

interpreter.responding = False
interpreter.computer
message = ""
last_message_type = None

for chunk in interpreter.chat(prompt, display=True, stream=True):  # type: ignore
    if PRINT_PARSING_DETAIL:
        print("[*] Reply:", chunk)
    try:
        chunk_typed = MessageChunk.parse_obj(chunk)
        if PRINT_PARSING_DETAIL:
            print("[*] Reply (parsed):", chunk_typed)
        if last_message_type is not None:
            if last_message_type != chunk_typed.type:
                if PRINT_MESSAGE_CHUNK:
                    print(f"[*] Message ({last_message_type})", message)
                message = ""

        if chunk_typed.content is not None:
            if type(chunk_typed.content) == str:
                message += chunk_typed.content
            elif type(chunk_typed.content) == MessageContent:
                if type(chunk_typed.content.content) == str:
                    message += chunk_typed.content.content
        last_message_type = chunk_typed.type

    except pydantic.ValidationError:
        traceback.print_exc()
        print(
            "[-] Failed to parse reply"
        )  # please use it as is. we do not know how deep the water is


if message != "":
    if PRINT_MESSAGE_CHUNK:
        print(f"[*] Message ({last_message_type})", message)
    message = ""
