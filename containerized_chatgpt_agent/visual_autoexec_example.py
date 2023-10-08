from port_util import port

# TODO: multi-agent infrastructure, help each other to earn bucks
# TODO: train the model on some 'visual' datasets
# TODO: diff/diffuse the input
# TODO: limit the output. prevent the ai from going too far (talkative), if it is not doing any valid operation
# TODO: make the model 'error-free', that is, interpreting & executing the output no matter what it is

urlbase = f"http://localhost:{port}"
import functools


@functools.lru_cache()
def urlmake(path):
    return f"{urlbase}/{path}"


import litellm
import requests

model_name = "ollama/autoexec_visual"

sess = requests.Session()


def perform_action(path: str, params: dict):
    url = urlmake(path)
    response = sess.get(url, params=params)
    return response


def get_info(path: str):
    response = perform_action(path, {})
    data = response.json()
    return data


def get_resolution():
    data = get_info("resolution")
    return data["width"], data["height"]


def get_position():
    data = get_info("position")
    return data["x"], data["y"]


def get_text_screenshot():
    data = get_info("text_screenshot")
    return data["text"]


def move_abs_action(argument: str):
    x, y = argument.split(",")
    x, y = x.strip(), y.strip()
    x, y = float(x), float(y)
    perform_action("move_abs", {"x": x, "y": y})


def type_action(argument: str):
    argument = unescape(argument)
    perform_action("type", {"text": argument})


def click_action(argument: str):
    button = argument.strip()
    params = {}
    if argument:
        params = {"button": button}
    perform_action("click", params)


action_handlers = {
    "move_abs": move_abs_action,
    "type": type_action,
    "click": click_action,
}

import random


def move_abs_random_action(width, height):
    x = random.randint(0, width)
    y = random.randint(0, height)
    action = f"move_abs {x},{y}"
    return action


import ast


def unescape(text: str):
    text = ast.literal_eval(repr(text).replace("\\\\", "\\"))
    return text


def escape(text: str):
    text = text.encode("unicode_escape").decode()
    return text


def type_random_action(min_char=4, max_char=10):
    text_length = random.randint(min_char, max_char)
    text = "".join(chr(random.randint(0, 255)) for _ in range(text_length))
    text = escape(text)
    action = f"type {text}"
    return action


def click_random_action():
    button_choices = ["left", "right", "middle", None]
    button = random.choice(button_choices)
    if button:
        action = f"click {button}"
    else:
        action = "click"
    return action


random_action_generators = {
    "move_abs": move_abs_random_action,
    "type": type_random_action,
    "click": click_random_action,
}


def random_actor(width, height, min_action=1, max_action=3):
    random_actions = []
    action_count = random.randint(min_action, max_action)
    for _ in range(action_count):
        action_name, action_generator = random.choice(
            list(random_action_generators.items())
        )
        args = []
        if action_name == "move_abs":
            args.extend([width, height])
        action = action_generator(*args)
        random_actions.append(action)
    return random_actions


import traceback


def action_executor(action_text: str):
    action_text = action_text.lstrip()
    err = None
    for action, handler in action_handlers.items():
        if action_text.startswith(action):
            argument = action_text[len(action) + 1 :]
            print("excuting:", action, argument)
            try:
                handler(argument)
            except:
                err = traceback.format_exc(limit=1)
                print("err:", err)
            break
    return err


# at the same time, how do we visualize the current display?
# you need to name that container.


def execute_command_list(cmd_list):
    err_list = []
    for cmd in cmd_list:
        err = action_executor(cmd)
        if err:
            err_list.append(err)
    return err_list


import time

SLEEP_TIME = 3


def construct_prompt(
    data: str, width: int, height: int, random_err_list: list[str], err_list: list[str]
):
    random_commands = random_actor(width, height)
    x, y = get_position()
    random_commands_str = "\n".join(random_commands)
    last_random_errors = "\n".join(random_err_list)
    last_errors = "\n".join(err_list)
    prompt = f"""
{data}

Pointer location: {x}, {y}

Resolution: {width}x{height}

Last random command errors:
{last_random_errors}

Last errors:
{last_errors}

Next random commands:
{random_commands_str}

Your commands:
"""
    return prompt, random_commands


def get_reply_from_chatgpt(content: str):
    messages = [{"content": content, "role": "system"}]
    print("sending:")
    print(messages)
    response = litellm.completion(
        model_name, messages, api_base="http://localhost:11434"
    )
    choices = response["choices"]
    reply_content = choices[0]["message"]["content"]
    print("reply:")
    print(reply_content)
    return reply_content


err_list = []
random_err_list = []
width, height = get_resolution()
while True:
    data = get_text_screenshot()
    prompt, random_commands = construct_prompt(
        data.strip(), width, height, random_err_list, err_list
    )
    print("random commands:", random_commands)
    response = get_reply_from_chatgpt(prompt)
    command_list = response.split("\n")
    random_err_list = execute_command_list(random_commands)
    err_list = execute_command_list(command_list)
    time.sleep(SLEEP_TIME)
