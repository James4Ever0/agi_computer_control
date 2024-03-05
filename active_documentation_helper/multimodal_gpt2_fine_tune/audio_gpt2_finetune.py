# check whisper
# load dataset and run the code in kaggle, then paste the code here.
from transformers import ASTFeatureExtractor, ASTModel
import torch

audio_array = torch.rand(1, 100000)

model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

sampling_rate = 16000

processor = ASTFeatureExtractor.from_pretrained(model_name)
model = ASTModel.from_pretrained(model_name)

inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

pooler_output = outputs["pooler_output"]