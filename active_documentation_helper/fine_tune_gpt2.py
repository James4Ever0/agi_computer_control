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


try:
    model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(
        "distilgpt2-godlang"
    )  # right from the model checkpoint save directory.

    tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(
        "distilgpt2-godlang", use_fast=True
    )
except:
    model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(
        model_checkpoint, use_fast=True
    )

    # TODO: change model embedding size

    tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
# tokenizer.add_special_tokens({"pad_token": "[PAD]"})

PAD_ID = tokenizer("<|endoftext|>")["input_ids"][0]  # type: ignore

# breakpoint()

datadict = {"input_ids": [], "labels": [], "attention_mask": []}

datadir = "gpt2_godlang_dataset"

# datadir = "godlang_dataset"

split_size = 128
import progressbar

# TODO: inverse embedding
INVERSE_TOKENIZER = True

for filename in progressbar.progressbar(os.listdir(datadir)[:10]):
    # for filename in progressbar.progressbar(os.listdir(datadir)):
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
        ret["labels"] = ret["input_ids"].copy()
        for k, v in ret.items():
            datadict[k].extend(v)


dataset = Dataset.from_dict(datadict)

train_dataset = dataset.shuffle().select(range(100))
# train_dataset = dataset.shuffle().select(range(100))
eval_dataset = dataset.shuffle().select(range(10))
# eval_dataset = dataset.shuffle().select(range(100))

training_args = TrainingArguments(
    f"./{model_checkpoint}-godlang",
    evaluation_strategy="epoch",
    # learning_rate=2e-5,  # negative learning rate
    learning_rate=2e-10,
    weight_decay=0.01,
    push_to_hub=False,  # Change to True to push the model to the Hub
    save_total_limit=1,
)
from transformers import TrainerCallback


import math

class StopTrainingOnNaNCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        for it in state.log_history:
            eval_loss = it.get("eval_loss")
            # if math.isnan(eval_loss):
            print("EVAL LOSS:", eval_loss)
            if eval_loss is not None and math.isnan(eval_loss):
                print("STOPPING TRAINING")
                control.should_training_stop = True
                break


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    callbacks=[StopTrainingOnNaNCallback()],
)


def check_nan(val):
    is_nan = math.isnan(val)
    return is_nan


trainer.train()

eval_results = trainer.evaluate()

print(f'Perplexity: {math.exp(eval_results["eval_loss"]):.2f}')
is_nan = check_nan(eval_results["eval_loss"])
print("loss nan?", is_nan)

if is_nan:
    print("model not saved")
else:
    trainer.save_model()
