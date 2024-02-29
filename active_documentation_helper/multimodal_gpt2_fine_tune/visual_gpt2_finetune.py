# load image patches, transform into image embeddings and then put along with text embeddings.

def image_to_patches(image): # did MiniGPT4 make small patches for images? or just resize them to fit the ViT input size?
    return patches # 224x224

def patches_to_embeddings(patches):
    return image_embeddings

def text_to_embeddings(text):
    return text_embeddings

def combine_embeddings(image_embeddings, text_embeddings):
    return embeddings

# try to replace field "input_ids" with "inputs_embeds" in dataset, but that will be bad since embeddings are huge.

# implement training logic yourself, or find existing visual & language model trainer on huggingface.

from transformers import ViTFeatureExtractor # this thing seems not doing any cropping process.