# mix godlang dataset with other dataset.
# one line godlang, one line other dataset.

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["https_proxy"] = ""
os.environ["http_proxy"] = ""
os.environ["all_proxy"] = ""

import requests.sessions

DEFAULT_HEADERS = dict(requests.sessions.default_headers())
DEFAULT_HEADERS["Accept-Encoding"] = "identity"


def new_default_headers():
    # print("Creating default headers")
    return requests.sessions.CaseInsensitiveDict(DEFAULT_HEADERS)


del requests.sessions.default_headers
setattr(requests.sessions, "default_headers", new_default_headers)

import datasets

dataset_name = "andersonbcdefg/synthetic_retrieval_tasks"

dat = datasets.load_dataset(dataset_name)

# print(dat) # ['task', 'iteration']
# breakpoint()

task = dat['train']['task']

import random

def get_random_task():
    return random.choice(task)

from godlang_fuzzer import CommandGenerator

def get_mixed_dataset(count:int):
    for _ in range(count):
        selector = random.randint(0,1)
        if selector == 0:
            command = get_random_task()
        else:
            command = CommandGenerator.call_single_random_command()
        print(command)

if __name__ == "__main__":
    get_mixed_dataset(10000)