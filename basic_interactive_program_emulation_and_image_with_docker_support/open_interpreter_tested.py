# patching input function
# from interpreter.core.default_system_message import default_system_message
import typing
import traceback
import re
import platform

PROMPT_PATH = "/tmp/prompt.txt"


def get_system_info():
    cmd = "neofetch --stdout".split()
    system_info = subprocess.check_output(cmd, encoding="utf-8")
    return system_info


def get_processor_architecture():
    ret = f"Processor architecture: {platform.processor()}"
    return ret


def get_python_version():
    ret = f"Python: {platform.python_version()}"
    return ret


def build_custom_system_prompt():
    ret = f"""
You are Open Interpreter, a world-class programmer that can complete any goal by executing code.
First, write a plan. **Always recap the plan between each code block** (you have extreme short-term memory loss, so you need to recap the plan between each message block to retain it).
When you execute code, it will be executed **on the user's machine**. The user has given you **full and complete permission** to execute any code necessary to complete the task. Execute the code.
You can access the internet. Run **any code** to achieve the goal, and if at first you don't succeed, try again and again.
You can install new packages.
When a user refers to a filename, they're likely referring to an existing file in the directory you're currently executing code in.
Write messages to the user in Markdown.
In general, try to **make plans** with as few steps as possible. As for actually executing code to carry out that plan, for *stateful* languages (like python, javascript, shell, but NOT for html which starts from 0 every time) **it's critical not to try to do everything in one code block.** You should try something, print information about it, then continue from there in tiny, informed steps. You will never get it on the first try, and attempting it in one go will often lead to errors you cant see.
You are capable of **any** task.

{get_system_info()}
{get_processor_architecture()}
{get_python_version()}
""".strip()
    return ret


def replace_sudo(multiline_string: str):
    updated_string = re.sub(r"^[ \t]*(sudo )", "", multiline_string, flags=re.MULTILINE)
    return updated_string


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
        super().__init__(*args, **kwargs)  # type: ignore
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

from interpreter.core.computer.terminal.languages.subprocess_language import (
    SubprocessLanguage,
)
import time


def subprocess_input(self: SubprocessLanguage, command: str, suffix: str = "\n"):
    print("[*] Writing to stdin:", command)
    self.process.stdin.write(command + suffix)  # type: ignore
    self.process.stdin.flush()  # type: ignore


def yes(self: SubprocessLanguage, interval=2):
    while True:
        # subprocess_input(self,"y")
        # subprocess_input(self,"")
        time.sleep(interval)


setattr(SubprocessLanguage, "input", subprocess_input)
old_start_process = copy.copy(SubprocessLanguage.start_process)


def new_start_process(self: SubprocessLanguage):
    old_start_process(self)
    print("[*] Starting yes input thread.")
    threading.Thread(target=yes, daemon=True, args=(self,)).start()


setattr(SubprocessLanguage, "start_process", new_start_process)

from interpreter.terminal_interface import terminal_interface

setattr(terminal_interface, "input", custom_input)

from interpreter import OpenInterpreter
from interpreter.core.computer.computer import Computer

SYSTEM_MESSAGE = build_custom_system_prompt()

interpreter = OpenInterpreter(
    disable_telemetry=True, system_message=SYSTEM_MESSAGE, loop=True
)
# interpreter = OpenInterpreter(disable_telemetry = True, system_message=default_system_message)


# old_run = copy.copy(computer.run)
class CustomComputer(Computer):
    def run(self, language, code, **kwargs):
        print("[*] Calling custom computer run method.")
        print(f"[*] Language: {language}")
        print("[*] Code:")
        print(code)
        print("[*] Kwargs:", kwargs)
        if language == "bash":
            print("[*] Removing sudo from bash code")
            code = replace_sudo(code)
            print("[*] Processed code:")
            print(code)
        ret = super().run(
            language, code, **kwargs
        )  # <generator object Terminal._streaming_run>
        print("[*] computer execution result:", ret)
        return ret


computer = CustomComputer(interpreter)
# setattr(computer, 'run', new_run)
interpreter.computer = computer


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

if os.path.isfile(PROMPT_PATH):
    print("[*] Reading prompt from:", PROMPT_PATH)
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt = f.read()
else:
    print("[*] Prompt file not found at:", PROMPT_PATH)
    prompt = "get me the ip address of bing.com."

print("[*] Using prompt:", prompt)
# prompt = "install nmap command."

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

# you can do generator exit here. the interpreter will handle it.
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

history_messages = interpreter.messages
print("[*] History messages:")
print(history_messages)
