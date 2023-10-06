
import litellm
import base64
import requests

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", type=int, default=8788, help="port number")
args = parser.parse_args()
port = args.port
assert port > 0 and port < 65535
print("using server on port %d" % port)

# it is bad to run random commands.
# maybe you should listen to the advice at https://github.com/Significant-Gravitas/AutoGPT/issues/346
# before it is too late.

# we are just prototyping. why so serious.
# trying random stuff!

# let's create a virtual editor.

# you just need to master the diff, the memory and the action

prompt_gen= lambda content: f"""
You are an AI agent inside a terminal environment. You can interact with the environment by writing special commands separated by newline. After your actions, the environment will execute the command and return the current terminal view. You can input special characters like carriage return or delete by using escape sequences, starting with a backslash and ending with a letter, like '\\b'.

Avaliable commands:

type <character sequence>

Example:

type echo hello world\\n
type date\\n

Terminal environment:

{content}

Your command:
"""

def get_terminal_data(port):
    r = requests.get(f"http://localhost:{port}/display")
    return r.text

def construct_prompt(data):
    prompt = prompt_gen(data)
    return prompt

model_tag = "openai/gpt-3.5-turbo"


def get_reply_from_chatgpt(content: str):
    messages = [{"content": content, "role": "user"}]
    print("sending:")
    print(messages)
    # openai call
    # many info inside. you may want to take a look?
    response = litellm.completion(model_tag, messages)
    choices = response['choices']
    reply_content = choices[0]['message']['content']
    print("reply:")
    print(reply_content)
    return reply_content


def parse_command_list(response):
    cmd_prefix = "type "
    command_list = []
    for line in response.split("\n"):
        if line.startswith(cmd_prefix):
            command = line[len(cmd_prefix):]
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

while True:
    data = get_terminal_data(port)
    prompt = construct_prompt(data.strip())
    response = get_reply_from_chatgpt(prompt)
    command_list = parse_command_list(response)
    execute_command_list(command_list, port)