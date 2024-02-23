# usage: python3.9 create_godlang_dataset.py --command_interval 1 --command_batch_size 5 --total_batches 100 --output_file output.txt

# TODO: add time token for our session like: Today, we<|time|> are going to <|time|> produce a new session<|time|>

# TIME_TOKEN_INSERT_INTERVAL = 1 # in seconds

import argparse
from typing import Callable

from godlang_fuzzer import CommandGenerator
from test_agent import main_template


def random_command_list_generator(command_batch_size: int):
    command_list: list[str] = []
    for _ in range(command_batch_size):
        command = CommandGenerator.call_single_random_command()
        command_list.append(command)
        print(command)
    return command_list


import asyncio
import functools


def create_dataset(command_list_generator: Callable[[int], list[str]]):
    parser = argparse.ArgumentParser()

    parser.add_argument("--command_interval", type=float, default=0.2)
    parser.add_argument("--command_batch_size", type=int, default=5)
    parser.add_argument("--total_batches", type=int, default=100)
    # parser.add_argument("--output_file", type=str, required=True)

    arguments = parser.parse_args()

    command_interval: int = arguments.command_interval
    command_batch_size: int = arguments.command_batch_size
    total_batches: int = arguments.total_batches
    # output_file: str = arguments.output_file

    asyncio.get_event_loop().run_until_complete(
        main_template(
            functools.partial(command_list_generator, command_batch_size),
            total_batches=total_batches,
            regular_sleep_time=command_interval,
        )
    )


if __name__ == "__main__":
    create_dataset(random_command_list_generator)
