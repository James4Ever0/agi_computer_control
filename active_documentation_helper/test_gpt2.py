import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["https_proxy"] = ""
os.environ["http_proxy"] = ""
os.environ["all_proxy"] = ""

model_dir = "distilgpt2-godlang"

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

from transformers import pipeline  # set_seed

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# no need to set seed.
# set_seed(42)

# it is been fucked up by random sequences.

from godlang_fuzzer import CommandGenerator

random_command = CommandGenerator.call_single_random_command()
print("init command:", repr(random_command))
response_list = generator(f"{random_command}\n", max_length=100, num_return_sequences=5)
# response_list = generator("VIEW\n", max_length=100, num_return_sequences=5)
# response_list = generator("Hello, I’m a language model", max_length=20, num_return_sequences=5)

# how do you plan to use it?
# would you like to change the godlang syntax?

# these are a few possibilities that the model can do. you would like to choose the most useful response as our dataset candidate, but how?
for response in response_list:
    for line in response["generated_text"].split("\n"):
        print(line)
    print("-" * 60)

# you would extract effective commands from response.