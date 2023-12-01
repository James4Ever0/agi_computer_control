import torch
import torch.nn as nn
import torch.nn.functional as F


class VKQAttention(nn.Module):
    def __init__(self, input_dim):
        super(VKQAttention, self).__init__()
        self.input_dim = input_dim
        self.value = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.query = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        self.dk = self.input_dim

    def forward(self, x):
        values = self.value(x)
        keys = self.key(x)
        queries = self.query(x)
        # scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.dk**0.5)
        # scores = torch.bmm(keys.transpose(1, 2), queries) / (self.dk**0.3)  # not 2!
        scores = torch.bmm(keys.transpose(1, 2), queries) / (self.dk**0.5)
        attention = self.softmax(scores)
        # print(attention)
        # print(scores)
        # breakpoint()
        # weighted = torch.bmm(scores, values)
        # weighted = torch.bmm(attention, values)
        weighted = torch.bmm(values, attention)
        return weighted


import torch

# import torchvision.models as models

from bnn import BConfig, prepare_binary_model

# Import a few examples of quantizers
from bnn.ops import BasicInputBinarizer, BasicScaleBinarizer, XNORWeightBinarizer


# Define the binarization configuration and assign it to the model
bconfig = BConfig(
    activation_pre_process=BasicInputBinarizer,
    activation_post_process=BasicScaleBinarizer,
    # optionally, one can pass certain custom variables
    weight_pre_process=XNORWeightBinarizer.with_args(center_weights=True),
)
# Convert the model appropiately, propagating the changes from parent node to leafs
# The custom_config_layers_name syntax will perform a match based on the layer name, setting a custom quantization function.
model = VKQAttention(input_dim=768)
bmodel = prepare_binary_model(model, bconfig)

inp = torch.randn(1, 12, 768)
# out = model(inp)
# print(out, out.shape)
# print(inp)
# print(out.shape)  # 1, 12, 768
bout = bmodel(inp)
print(bout)
# print(bout.shape)
