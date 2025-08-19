LLM shall be pretrained on large scale hexadecimal data before required to do program synthesis.

We can encode every hex byte in its ascii form into a unique token. We can markup the hexadecimal area with special tokens. If the tokenizer supports raw bytes input and output, we can map the real binary byte into token id.

We can prepare such a python script and dataset for LLM to train on some binary programs and do predictions.

---

I had an idea that programs shall ingest programs and learn to evolve in a evolutionary manner.

The program shall directly execute assembly code. So obviously, no human interface is needed.

You may ask how do we interact with such program? Consider computer virus. We first run it in isolated environments, then we interact.

---

you may make hierarchical tokenizer or hierarchical embedding to reduce memory consumption
