# load image patches, transform into image embeddings and then put along with text embeddings.

# try to replace field "input_ids" with "inputs_embeds" in dataset, but that will be bad since embeddings are huge.

# implement training logic yourself, or find existing visual & language model trainer on huggingface.

import numpy as np
from empatches import EMPatches
import torch
from transformers import ViTFeatureExtractor, ViTModel

image = np.random.rand(448, 448, 3)

emp = EMPatches()

patches, indices = emp.extract_patches(image, patchsize = 224, overlap = 0)

model_name = "google/vit-base-patch16-224-in21k"

model = ViTModel.from_pretrained(model_name) # if with pooling, you can get `outputs.pooler_output`.
processor = ViTFeatureExtractor.from_pretrained(model_name)

inputs = processor(patches, do_rescale=False) # setting "do_rescale" to False (default to True) is crucial for images with values in range 0 to 1

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state[:, 0, :]