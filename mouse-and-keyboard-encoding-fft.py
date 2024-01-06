#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

# mouse -> sparse encoding -> fft -> ifft -> unified decoder

dim_0_range = 1000
dim_1_range = 100

mouse_coords = [(20,20,None,None), (200,200,30,30)]


# In[2]:


# what about sparse encoding?

# single value -> bunch of binary values
# elementwise product random vector -> select non-zero ones

import random

random.seed(42)

window_size = 200 

mrange = list(range(window_size+dim_0_range-1))

random.shuffle(mrange)

# mlist = [mrange[i:i+window_size] for i in range(dim_0_range)]

# keep it sparse?

unified_encoding = torch.randn((1,window_size+dim_0_range-1), requires_grad=True)

# that's how you initialize your "semantic" or "continual" mouse embeddings.

mlist = []

next_comb = mrange[:window_size]
# random.shuffle(next_comb)

mlist.append(next_comb.copy())

for i in range(dim_0_range-1):
#     last_item = mrange[i+window_size-1]
    alt_item = mrange[i+window_size]
    last_item_index = random.choice(range(window_size))
    next_comb[last_item_index] = alt_item
#     print(torch.Tensor([next_comb]))
#     print("SUM?", sum(next_comb))
    mlist.append(next_comb.copy())

# import rich
print("LENGTH?", len(mlist))
# rich.print("MLIST?", mlist)

# for e in mlist:
#     print(sum(e))

mLongTensorList = torch.LongTensor(mlist) # that looks like the thing.
mLongTensorList.shape


# In[3]:


torch.index_select(unified_encoding, 1, mLongTensorList[0,:])


# In[4]:


# how to represent keyboard keydown signals?
# telephone?

# embedding plus one trainable sin keydown signal? or using fft?

# how to represent special tokens? by sin? all by sin?

# what will happen if you try to share vector space in non-standard way?
# such as split and concat?

# you may do split and concat in fft though.

# such as: value repr by concat -> ifft -> LSTM -> fft -> argmax things

# ifft let the model "feel" the bits, though fft "extract" freq and handle bits.

# there are multiple ways to do this.

# but fft brings "imaginary" part to numbers.
# you can feed both parts separetely into the network, then combine different parts.
# the you may calculate the grad? by adding different part of the loss?

# or you use "native" complex neural networks, to handle the fft transforms.
# or you simply ignore complex input. only taking real parts?

# lstm contains hidden state and cell state
# while GRU only contains hidden state.

# adding real and imag? or passing through different NN? or same NN?
# telling you, do it first. we will handle the comparison.

# so are you going to tell me that my model is just complete
# that i need not to do too much to collect data and start training?

# yes. i am going to tell you to start training.
# you have reached the utopia of fourier transform (rfft/hfft).
# now let's roll!


# In[ ]:


# for ubuntu arm: pyautogui -> python script writing timestamp
# write to stdout -> pipe to ffmpeg -> write to video

# find the location of the shared directory of utm

# how do i know if my model is spitting words instead of images/actions?
# do we need to take over few "positional encodings"?

# you can add task specific embeddings with token embedding
# then decode the task type in the end, classify the token.

# fft may not be needed, since that will be too much computation.
# you may just want low rank adaption over some linear layers.

# fft may be useful for your visual convolution.

