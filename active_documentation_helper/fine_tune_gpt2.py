import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["https_proxy"] = ""
os.environ["http_proxy"] = ""
os.environ["all_proxy"] = ""
# os.environ['REQUESTS_CA_BUNDLE'] = ''

import requests.sessions

DEFAULT_HEADERS = dict(requests.sessions.default_headers())
DEFAULT_HEADERS["Accept-Encoding"] = "identity"


def new_default_headers():
    # print("Creating default headers")
    return requests.sessions.CaseInsensitiveDict(DEFAULT_HEADERS)


del requests.sessions.default_headers
setattr(requests.sessions, "default_headers", new_default_headers)

# from huggingface_hub import list_datasets

# print([dataset.id for dataset in list_datasets()])

# from datasets import load_dataset

# dat = load_dataset("squad")
# breakpoint()

# import requests

# r = requests.get("https://hf-mirror.com/api/datasets")
# # r = requests.get("https://hf-mirror.com/api/datasets", headers=DEFAULT_HEADERS)
# print(r.text)

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

model_checkpoint = "distilgpt2"
tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(
    model_checkpoint, use_fast=True
)

model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained("distilgpt2")

tokenizer.add_special_tokens({"pad_token": "[PAD]"})

PAD_ID = tokenizer("[PAD]")["input_ids"][0]

datadict = {"input_ids": [], "labels": [], 'attention_mask': []}

datadir = "godlang_dataset"
split_size = 128
import progressbar

for filename in progressbar.progressbar(os.listdir(datadir)):
    with open(
        os.path.join(datadir, filename), "r"
    ) as f:  # maybe it is impossible to decode some file.
        content = f.read()
        ret = tokenizer(content)
        ret = {
            k: v + ([PAD_ID] * ((split_size - len(v) % split_size) % split_size))
            for k, v in ret.items()
        }
        ret = {
            k: [v[i : i + split_size] for i in range(0, len(v), split_size)]
            for k, v in ret.items()
        }
        ret['labels'] = ret['input_ids'].copy()
        for k, v in ret.items():
            datadict[k].extend(v)


dataset = Dataset.from_dict(datadict)


print(dataset[0])  # it is been truncated.
breakpoint()
