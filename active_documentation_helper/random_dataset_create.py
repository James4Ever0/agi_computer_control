# create random dataset from available tokens in gpt2 tokenizer
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["https_proxy"] = ""
os.environ["http_proxy"] = ""
os.environ["all_proxy"] = ""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
# print(dir(tokenizer))
# breakpoint()

vocab = list(tokenizer.vocab.keys())
# breakpoint()
import random

def get_one_token():
    return random.choice(vocab)

def get_multiple_tokens(count:int):
    candidates = []
    for _ in range(count):
        candidates.append(get_one_token())
    return "".join(candidates)

if __name__ == "__main__":
    count = 10000
    print(get_multiple_tokens(count))