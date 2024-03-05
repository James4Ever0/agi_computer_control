#!/usr/bin/env python
# coding: utf-8

# # Multimodal Transformer based on GPT2

# ## Load the GPT2 model

# In[2]:


import transformers


model_name = "distilgpt2"
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs, labels=inputs["input_ids"])
# loss = outputs.loss

print(model)


# ## Encode tokens into embeddings

# In[23]:


# dir(model)

input_ids = inputs['input_ids'] # 6 tokens
input_embeds = model.transformer.wte(input_ids)

# print("embed_dim:",model.base_model.embed_dim) # 768
# model(inputs_embeds=input_embeds, attention_mask=inputs['attention_mask'], labels=inputs['input_ids'])

print(inputs['input_ids'].shape) # [1,6]
print(input_embeds.shape) # [1,6,768]


# ## Generate video embeddings

# In[18]:


from transformers import AutoImageProcessor, ViTModel
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)


# In[24]:


type(image)


# In[74]:


random_image = torch.rand(3, 224, 224)
image_processor(random_image, do_rescale=False, return_tensors="pt") # from official doc.


# In[72]:


import numpy
import numpy as np
# torch_random_image = numpy.random.rand(3, 224, 224)
torch_random_image = np.random.randint(0, 256, (224, 224, 3), dtype=numpy.uint8)

# torch_random_image

# 480, 640, 3, uint8
# print(image_processor(torch_random_image,return_tensors="pt")['pixel_values'].shape) # negative values
# print(image_processor(torch_random_image,return_tensors="pt")['pixel_values'].mean())
# print(torch_random_image.mean())
# model(**image_processor(torch_random_image,return_tensors="pt"))
numpy.array(image).shape, image_processor(torch_random_image, return_tensors="pt"), torch_random_image, image_processor(image)


# In[2]:


print([key for key in dir(outputs) if not key.startswith('_')])


# In[8]:


# outputs.attentions # attention tensor per layer.
outputs.attentions[-1].shape # [1, 12, 197, 197]


# In[8]:


outputs.hidden_states[-1].shape # [1, 197, 768]


# In[79]:


outputs.last_hidden_state[:,0,:].shape


# In[40]:


# dir(outputs)
# [batch, channel, height, width]
# print(inputs['pixel_values'].shape) # torch.Size([1, 3, 224, 224])
# inputs['pixel_values']
print(inputs['pixel_values'].mean())


# In[ ]:


# load ViT model from huggingface.


# ## Generate audio embeddings

# In[7]:


# now, audio tokens.

# load audio dataset, or just use your own random data.

# load audio feature extractor, "AST" model, or WhisperFeatureExtractor

# load model for audio classification or ASTModel.

from transformers import AutoProcessor, ASTModel
import torch
from datasets import load_dataset

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# audio file is decoded on the fly
inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

# modify the length of the input

inputs.input_values=inputs.input_values[:,:512,:]

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

# the input state will be padded, if it is too short.
list(last_hidden_states.shape),list(inputs.input_values.shape) # [1, 1214, 768]


# In[10]:


# outputs.keys() # ordered dict, ['last_hidden_state', 'pooler_output']

outputs['pooler_output'].shape # [1, 768], ready for audio embedding and classification


# In[9]:


outputs.last_hidden_state.shape # torch.Size([1, 1214, 768])


# In[117]:


type(processor), type(model)


# In[129]:


dataset[0]['audio']['array'].shape, inputs.input_values.shape 

# 1024 is the max length of the fbank transform.


# ((93680,), torch.Size([1, 1024, 128]))
# independent of audio length? 


# In[ ]:


# accepts at most 10.24 seconds of audio.
# so we can deduce: max_length/100 = max_audio_length_in_seconds

# window_shift = int(sample_frequency * frame_shift / 1000)
# m = num_samples / window_shift (approximately)

# to ensure consistency you might want to pad or truncate fbank output 

# PS: WhisperFeatureExtractor uses chunk_length=30 to limit input to 30 seconds.
# Whisper model is bounded to 30 second inputs. Shorter inputs need to be padded.


# In[133]:


# get fbank actual length

import torchaudio

waveform = torch.from_numpy(dataset[0]['audio']['array']).unsqueeze(0)
sample_frequency = processor.sampling_rate
num_mel_bins = processor.num_mel_bins

fbank = torchaudio.compliance.kaldi.fbank(waveform, sample_frequency = sample_frequency, window_type='hanning', num_mel_bins = num_mel_bins)
fbank.shape # torch.Size([584, 128])


# In[ ]:


# to run multilingual transcription, first slice the audio by speakers, then run whisper on each segment
# https://github.com/pyannote/pyannote-audio (speaker diarization)

# For audio segmentation and classification, check You-Only-Hear-Once
# So we can differentiate speech, music or noise

