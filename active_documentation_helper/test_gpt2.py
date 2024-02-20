import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["https_proxy"] = ""
os.environ["http_proxy"] = ""
os.environ["all_proxy"] = ""

model_dir = "distilgpt2-godlang"

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

from transformers import pipeline, set_seed

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

set_seed(42)

# it is been fucked up by random sequences.

response_list = generator("VIEW\n", max_length=100, num_return_sequences=5)
# response_list = generator("Hello, I’m a language model", max_length=20, num_return_sequences=5)

for response in response_list:
    print(response)