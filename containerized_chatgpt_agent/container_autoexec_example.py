# TODO: replace this unstable api with another
# TODO: make sure the terminal service is always alive, sometimes the service is not responsive because of commands like ">8"
# TODO: insert & execute random commands
# TODO: build multi-agent framework
# TODO: memory framework, cache & permanent storage
# TODO: use image to ascii protocol (with ocr) for gui manipulation

# llama2 is not intelligent enough to complete this task.
# still, we can build framework upon this.


import ast


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
    rng = lambda: random.randint(0,255)
    for _ in range(random.randint(_min, _max)):
        cmd += chr(rng())
    cmd = escape(cmd)
    return cmd_prefix+cmd

def random_command_generator(_min = 5, _max = 10, min_count=1, max_count=3):
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
    prompt = f"""
Terminal environment:

{content}

Random commands:

{random_command_repr}

Your commands:
"""
    return prompt
def get_terminal_data(port):
    r = requests.get(f"http://localhost:{port}/display")
    return r.text

def construct_prompt(data):
    random_command_list = random_command_generator()
    prompt = prompt_gen(data, random_command_list)
    return prompt, random_command_list

# model_tag = "openai/gpt-3.5-turbo"

# import functools

# @functools.lru_cache(maxsize=100)
def get_reply_from_chatgpt(content: str):
    messages = [{"content": content, "role": "system"}]
    # messages = [{"content": content, "role": "user"}]
    print("sending:")
    print(messages)
    # openai call
    # many info inside. you may want to take a look?
    # response = litellm.completion(f"openrouter/{openrouter_model_name}", messages)
    # response = litellm.completion("ollama/llama2", messages, api_base="http://localhost:11434")
    response = litellm.completion("ollama/autoexec", messages, api_base="http://localhost:11434")
    choices = response['choices']
    reply_content = choices[0]['message']['content']
    print("reply:")
    print(reply_content)
    return reply_content

import ast

def parse_command_list(response):
    command_list = []
    for line in response.split("\n"):
        line = line.lstrip()
        if line.startswith(cmd_prefix):
            command = line[len(cmd_prefix):]
        # command = ast.literal_eval(repr(line).replace("\\\\","\\"))
            command = unescape(command)
            command_list.append(command)
    return command_list

def execute_command(command, port):
    print("executing command:", repr(command))
    b64command = base64.b64encode(command.encode('utf-8')).decode('utf-8')
    params = dict(b64type= b64command)
    requests.get(f"http://localhost:{port}/input", params=params)

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
    prompt, random_commands = construct_prompt(data.strip())
    print("random commands:", random_commands)
    response = get_reply_from_chatgpt(prompt)
    execute_command_list(random_commands, port)
    command_list = parse_command_list(response)
    execute_command_list(command_list, port)
    time.sleep(SLEEP_TIME)