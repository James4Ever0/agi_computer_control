# TODO: replace this unstable api with another
# TODO: make sure the terminal service is always alive, sometimes the service is not responsive because of commands like ">8"
# TODO: insert & execute random commands
# TODO: build multi-agent framework
# TODO: memory framework, cache & permanent storage
# TODO: use image to ascii protocol (with ocr) for gui manipulation
# TODO: get terminal size
# TODO: specify which part of response is executed, and which is not
# TODO: match error info with commands which cause problems
# TODO: do not clear the previous command execution records, instead keep a limited few and refresh
# TODO: create some interface to describe what commands does, or narriation.
import ollama_utils


# llama2 is not intelligent enough to complete this task.
# still, we can build framework upon this.

from terminal_config import cols, rows
import ast

generate_command_pool = lambda: {
    "executed": [],
    "not_executed": [],
    "error": [],
}


def refresh_command_pool(command_pool, limit=3):
    ret = {}
    for k, v in command_pool.items():
        new_v = v[-limit:]
        ret[k] = new_v
    return ret


prev_command_pool = generate_command_pool()


def unescape(text: str):
    text = ast.literal_eval(repr(text).replace("\\\\", "\\"))
    return text


def escape(text: str):
    text = text.encode("unicode_escape").decode()
    return text


import litellm
import base64
import requests
from port_util import port

print("using server on port %d" % port)

openrouter_model_name = "mistralai/mistral-7b-instruct"
cmd_prefix = "type "
import random


def generate_single_random_command(_min, _max):
    cmd = ""
    rng = lambda: random.randint(0, 255)
    for _ in range(random.randint(_min, _max)):
        cmd += chr(rng())
    cmd = escape(cmd)
    return cmd_prefix + cmd


def random_command_generator(_min=5, _max=10, min_count=1, max_count=3):
    count = random.randint(min_count, max_count)
    cmdlist = []
    for _ in range(count):
        cmd = generate_single_random_command(_min, _max)
        cmdlist.append(cmd)
    return cmdlist


# it is bad to run random commands.
# maybe you should listen to the advice at https://github.com/Significant-Gravitas/AutoGPT/issues/346
# before it is too late.

# we are just prototyping. why so serious.
# trying random stuff!

# let's create a virtual editor.

# you just need to master the diff, the memory and the action


def prompt_gen(content, random_command_list):
    random_command_repr = "\n".join(random_command_list)

    previous_executed_repr = "\n".join(prev_command_pool["executed"])
    previous_error_repr = "\n".join(prev_command_pool["error"])
    previous_not_executed_repr = "\n".join(prev_command_pool["not_executed"])

    prompt = f"""
Terminal environment:

{content}

Terminal size: {cols}x{rows}

Previous executed successfully:

{previous_executed_repr}

Previous executed with error:

{previous_error_repr}

Previous not executed:

{previous_not_executed_repr}

Random commands:

{random_command_repr}

Your commands:
"""
    return prompt


from diff_utils import diff_methods
from typing import Literal

prev_terminal_content = ""


def get_terminal_data(
    port,
    method: Literal[
        "git_style_diff", "char_diff", "line_indexed_diff", "no_diff"
    ] = "line_indexed_diff",
):
    global prev_terminal_content
    r = requests.get(f"http://localhost:{port}/display")
    terminal_content = r.text.strip()
    procedure = diff_methods.get(method, lambda prev, _next: _next)
    result = procedure(prev_terminal_content, terminal_content)
    prev_terminal_content = terminal_content
    return result


EXEC_DELAY = 0.5


def construct_prompt(data):
    random_command_list = random_command_generator()
    prompt = prompt_gen(data, random_command_list)
    return prompt, random_command_list


# model_tag = "openai/gpt-3.5-turbo"

# import functools


# @functools.lru_cache(maxsize=100)
def get_reply_from_chatgpt(content: str, max_tokens=50):
    # why you talk back to me! why are you so talktive!
    messages = [{"content": content, "role": "system"}]
    # messages = [{"content": content, "role": "user"}]
    print("sending:")
    print(messages)
    # openai call
    # many info inside. you may want to take a look?
    # response = litellm.completion(f"openrouter/{openrouter_model_name}", messages)
    # response = litellm.completion("ollama/llama2", messages, api_base="http://localhost:11434")
    response = litellm.completion(
        "ollama/autoexec",
        messages,
        api_base="http://localhost:11434",
        max_tokens=max_tokens,
    )
    choices = response["choices"]
    reply_content = choices[0]["message"]["content"]
    print("reply:")
    print(reply_content)
    return reply_content


import ast


def parse_command_list(response):
    command_list = []
    for _line in response.split("\n"):
        line = _line.lstrip()
        if line.startswith(cmd_prefix):
            command = line[len(cmd_prefix) :]
            # command = ast.literal_eval(repr(line).replace("\\\\","\\"))
            command = unescape(command)
            command_list.append(command)
            prev_command_pool["executed"].append(_line)
        else:
            prev_command_pool["not_executed"].append(_line)
    return command_list


def execute_command(command, port):
    print("executing command:", repr(command))
    b64command = base64.b64encode(command.encode("utf-8")).decode("utf-8")
    params = dict(b64type=b64command)
    requests.get(f"http://localhost:{port}/input", params=params)
    time.sleep(EXEC_DELAY)


def execute_command_list(command_list, port):
    print("total commands:", len(command_list))
    for command in command_list:
        execute_command(command, port)


import os

print("env:", os.environ)

import time

SLEEP_TIME = 3

while True:
    data = get_terminal_data(port)
    prompt, random_commands = construct_prompt(data)
    prev_command_pool = generate_command_pool()
    # prev_command_pool = refresh_command_pool(prev_command_pool)
    print("random commands:", random_commands)
    response = get_reply_from_chatgpt(prompt)
    prev_command_pool["executed"].extend(random_commands)
    execute_command_list([c[len("type ") :] for c in random_commands], port)
    command_list = parse_command_list(response)
    execute_command_list(command_list, port)
    time.sleep(SLEEP_TIME)
