#!/usr/bin/env python
# coding: utf-8

# In[34]:


# a good reference:
# https://blog.paperspace.com/generating-text-summaries-gpt-2/
get_ipython().system('pip3 install einops')

import einops
import transformers
import torch

MODEL_NAME = 'gpt2'
model = transformers.GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

lr = 0.0001
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

def build_special_token(location:str, name:str):
    return f"<|{location}_{name}|>"

def generate_special_token_pair(name:str):
    begin_token = build_special_token('begin', name)
    end_token = build_special_token('end', name)
    return begin_token, end_token

text = "8e7d4f"
# text = "0100100010010"
enc = tokenizer([text], return_tensors='pt')
input_ids = enc['input_ids'] # into three pieces only.
attention_mask = torch.ones(input_ids.shape, dtype=torch.int64)
input_ids.shape

begin_bytes, end_bytes = generate_special_token_pair('bytes')
# how to step in and out?
tokenizer(begin_bytes)['input_ids'], tokenizer(end_bytes)['input_ids']


# In[35]:


# print(dir(tokenizer))
# print(tokenizer.all_special_tokens)
# help(tokenizer.add_tokens)
tokenizer.add_tokens([begin_bytes, end_bytes]) # will not do this again.
# tokenizer.add_special_tokens({"begin_bytes": begin_bytes, "end_bytes":end_bytes})


# In[36]:


# add new special token to tokenizer
len(tokenizer)


# In[37]:


model.resize_token_embeddings(len(tokenizer))


# In[38]:


# dir(tokenizer)
# tokenizer.vocab

# binary_vocab = {i: format(i, '04b') for i in range(16)}
# binary_map = {v: tokenizer.vocab[v] for _, v in binary_vocab.items()}

# missing: 0011

hex_vocab = {i: format(i, '0x') for i in range(16)}
hex_map = {v: tokenizer.vocab[v] for _, v in hex_vocab.items()}
# hex_map


# In[39]:


byte_vocab = {i: str(i) for i in range(256)}
byte_map = {v: tokenizer.vocab[v] for _, v in byte_vocab.items()}


# In[46]:


output.logits.shape # now: 50259
# <|begin_bytes|>feffd3d7ea<|end_bytes|>
# <|begin_hex|>feffd3d7ea<|end_hex|>
# .............##########[#..........] <- in training we only mask some prob
# .............####################### <- in inference/parsing there could be state rolling back


# In[41]:


# training
output = model(input_ids = input_ids, attention_mask = attention_mask)
# output.logits[:,:,:] = 0 # this will not affect loss
masked_logits = torch.zeros(output.logits.shape)
focused_ids = [10,20,30]
masked_logits[:,:,focused_ids] = output.logits[:,:,focused_ids] # this will

zero_input_ids = torch.zeros(input_ids.shape, dtype=input_ids.dtype)
# output.logits
reshaped_original_logits = einops.rearrange(output.logits, "b s c -> b c s")
reshaped_logits = einops.rearrange(masked_logits, "b s c -> b c s")
loss = loss_fn(reshaped_original_logits, zero_input_ids)
# loss = loss_fn(reshaped_logits, zero_input_ids)
print(loss.item()) # it would be the same as long as setting to zero.


# In[42]:


masked_logits[:,:,focused_ids]


# In[43]:


model.zero_grad()


# In[44]:


loss.backward()
optimizer.step()
model.zero_grad()


# In[45]:


# inference

