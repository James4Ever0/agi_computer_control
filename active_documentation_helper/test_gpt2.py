import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["https_proxy"] = ""
os.environ["http_proxy"] = ""
os.environ["all_proxy"] = ""

model_dir = "distilgpt2-godlang"

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})

model = AutoModelForCausalLM.from_pretrained(model_dir)

from transformers import pipeline  # set_seed

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

command_prefix_list = ["TYPE", "VIEW", "WAIT", "REM", "SPECIAL"]


def has_command_prefix(cmd: str):
    for command_prefix in command_prefix_list:
        if cmd.startswith(command_prefix):
            return True
    return False


# no need to set seed.
# set_seed(42)

# it is been fucked up by random sequences.

from godlang_fuzzer import CommandGenerator


# response_list = generator(f"{random_command}\n", max_length=100, num_return_sequences=5)

# response_list = generator("VIEW\n", max_length=100, num_return_sequences=5)
# response_list = generator("Hello, I’m a language model", max_length=20, num_return_sequences=5)

# how do you plan to use it?
# would you like to change the godlang syntax?

# these are a few possibilities that the model can do. you would like to choose the most useful response as our dataset candidate, but how?

# TODO: use prefix 'SPECIAL' to completely standardize the language syntax, so that we can get those chat statements out

# you would like to create dataset from it.


def gpt2_command_generator(command_batch_size: int) -> list[str]:
    random_command = CommandGenerator.call_single_random_command()

    print("init command:", repr(random_command))
    response_list:list = generator(
        f"{random_command}\n", max_length=100, num_return_sequences=5
    ) # type:ignore
    response_list.sort(key=lambda x: -len(x["generated_text"]))

    # for response in response_list:
    response = response_list[0]
    # response = random.choice(response_list)

    gpt2_commands = response["generated_text"].split("\n")
    # you need to filter out those lines that are not commands
    selected_commands = []

    for cmd in gpt2_commands:
        if has_command_prefix(cmd):
            selected_commands.append(cmd)

    selected_commands = selected_commands[command_batch_size:]
    print("\n".join(selected_commands))
    return selected_commands


# print("-" * 60)
# so we would only have 5 commands each, sampled from the model.

# you would extract effective commands from response.

from create_godlang_dataset import create_dataset

if __name__ == "__main__":
    if os.environ.get("TEST", None) is not None:
        print("Testing")
        gpt2_command_generator(5)
    else:
        create_dataset(gpt2_command_generator)
