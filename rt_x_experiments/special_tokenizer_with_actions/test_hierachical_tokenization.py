# bytes -> char -> "token" -> sentences -> paragraphs
# what to do in reverse? must recurse into bytes.
# how to encode and decode?

from torch import nn
from torch import Tensor
import torch
from typing import Literal


# Convert bytes to binary representation as a list of integers using bitwise shift
def bytes_to_binary_int(byte_string):
    binary_representation = [
        int((byte_string[i] >> j) & 1)
        for i in range(len(byte_string))
        for j in range(7, -1, -1)
    ]
    return binary_representation


# Example usage
byte_string = b"hello world"
binary_representation_int = bytes_to_binary_int(byte_string)
print(binary_representation_int)

# you can pad with zeros when using bitlevel tokenizer
# but how do you shift? char level or bit level? neither? self-determined?

# encode:

# nx2 * 2*d -> n*d
# (nxn = nxd * dxn) * nxd = nxd
# nxd -> convolution -> mxd, m = n/2
# (mxm = mxd * dxm) * mxd = mxd

# decode:
# mxd -> deconvolution -> nxd
# nxd * dx2 -> nx2

# first, pad the output ahead of time.


# # so for every bit in the byte, we have a vector of 768 dimensions
# # try to reduce the bytes.

# we've got four level of abstractions.

from pydantic import BaseModel


class TransformerArguments(BaseModel, arbitrary_types_allowed=True):
    key_padding_mask: Tensor | None = None
    need_weights: bool = True
    attn_mask: Tensor | None = None
    average_attn_weights: bool = True
    is_causal: bool = False


class MultiheadSelfAttentionStack(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, **kwargs):
        super(MultiheadSelfAttentionStack, self).__init__()
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=embed_dim, num_heads=num_heads, **kwargs
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input_tensor: Tensor, transformerArguments: TransformerArguments):
        output = input_tensor
        for layer in self.layers:
            output, _ = layer(
                query=output, key=output, value=output, **transformerArguments.dict()
            )
        return output


# what if i use the inverse causal mask? backtracking mask?

# import torch
# a = torch.tril(torch.ones(10, 10))  # Create a lower triangular mask of ones
# b = torch.tril(a.T)  # Transpose the mask and then take the lower triangular part to ensure backtracking
# c = torch.tril(torch.flip(a, dims=[1]))  # Flip the mask along the horizontal axis and take the lower triangular part
# d = torch.tril(torch.flip(a, dims=[0]))  # Flip the mask along the vertical axis and take the lower triangular part

import einops
import torch.nn.functional as F


# hourglass replicate? not exactly. this is binary.
# what about moe? lsm?
class HierachicalTokenizationTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, num_layers=1, abstraction_level=4):
        # of course this is causal.
        super().__init__()
        self.TWO = 2
        self.ONE = 1
        self.ZERO = 0
        self.binary_embedding = nn.Embedding(
            num_embeddings=self.TWO, embedding_dim=embed_dim
        )

        self.abstractionLayers = []
        self.deabstractionLayers = []

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.abstraction_level = abstraction_level
        assert abstraction_level > 0, "abstraction level must be greater than zero"

        self.mainTransformer = MultiheadSelfAttentionStack(
            embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers
        )

        for _ in range(abstraction_level - 1):
            # for _ in range(abstraction_level):
            # Create the attention and abstraction layers
            att_layer = MultiheadSelfAttentionStack(
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers
            )
            abstract_layer = nn.Linear(self.TWO, self.ONE)
            self.abstractionLayers.append(
                att_layer
            )  # Add the attention layer to the list
            self.abstractionLayers.append(
                abstract_layer
            )  # Add the abstraction layer to the list

            # Create the inverse attention and deabstraction layers
            datt_layer = MultiheadSelfAttentionStack(
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers
            )
            deabstract_layer = nn.Linear(self.ONE, self.TWO)
            self.deabstractionLayers.append(
                datt_layer
            )  # Add the inverse attention layer to the list
            self.deabstractionLayers.append(
                deabstract_layer
            )  # Add the deabstraction layer to the list

        self.decode_embedding = nn.Linear(self.embed_dim, self.TWO)
        self.pad_size = self.TWO**abstraction_level

    def _sparseTransformerForwardImpl(
        self,
        embedding: Tensor,
        transformer: MultiheadSelfAttentionStack,
        transformerArguments: TransformerArguments,
    ):
        embedding = einops.rearrange(embedding, "b (s1 g) d -> (b g) s1 d", g=self.TWO)
        embedding = transformer(embedding, transformerArguments=transformerArguments)
        embedding = einops.rearrange(embedding, "(b g) s1 d -> b (s1 g) d", g=self.TWO)
        return embedding

    def sparseTransformerForward(
        self,
        embedding: Tensor,
        transformer: MultiheadSelfAttentionStack,
        transformerArguments: TransformerArguments,
        use_sliding=True,
    ):
        _embedding = self._sparseTransformerForwardImpl(
            embedding, transformer, transformerArguments
        )
        if use_sliding:
            slide_embedding = self.sparseTransformerForward(
                embedding[:, 1:-1, :],
                transformer,
                transformerArguments,
                use_sliding=False,
            )
            _embedding[:, 1:-1, :] = (_embedding[:, 1:-1, :] + slide_embedding) / 2
        return _embedding

    def calculateInputPadSizeFromSequenceLength(self, sequence_length: int):
        input_pad_size = []
        msequence_length = int(sequence_length)
        for _ in range(self.abstraction_level):
            div, mod = divmod(msequence_length, self.TWO)
            if mod == self.ONE:
                input_pad_size.append(self.ONE)
                div += self.ONE
            else:
                input_pad_size.append(self.ZERO)
            msequence_length = div
        return input_pad_size

    def padEmbedding(self, embedding: Tensor, pad_direction: Literal["left", "right"]):
        embedding = F.pad(
            embedding,
            (self.ZERO, self.ZERO, self.ZERO, self.ONE)
            if pad_direction == "right"
            else (self.ZERO, self.ZERO, self.ONE, self.ZERO),
            "constant",
            self.ZERO,
        )
        return embedding

    def chopEmbedding(self, embedding: Tensor, pad_direction: Literal["left", "right"]):
        embedding = (
            embedding[:, : -self.ONE, :]
            if pad_direction == "right"
            else embedding[:, self.ONE :, :]
        )
        return embedding

    def abstractionForward(self, embedding: Tensor, abstractionLayer: nn.Linear):
        embedding = einops.rearrange(embedding, "b (s1 g) d -> b s1 d g", g=self.TWO)
        embedding = abstractionLayer(embedding)  # Apply attention and abstraction
        embedding = einops.rearrange(embedding, f"b s d {self.ONE} -> b s d")
        return embedding

    def deabstractionForward(self, embedding: Tensor, deabstractionLayer: nn.Linear):
        embedding = einops.rearrange(embedding, f"b s d -> b s d {self.ONE}")
        embedding = deabstractionLayer(embedding)
        embedding = einops.rearrange(embedding, "b s1 d g -> b (s1 g) d", g=self.TWO)
        return embedding

    def forward(
        self,
        input_logits: Tensor,
        pad_direction: Literal["left", "right"],
        transformerArguments: TransformerArguments,
    ):  # it is trimmed from one side. is it causal?
        assert (
            len(input_logits.shape) == self.TWO
        ), "input logits shall be of shape (batch_size, sequence_length)"
        _, sequence_length = input_logits.shape
        assert sequence_length != self.ZERO, "zero length sequence encountered"
        input_pad_size = self.calculateInputPadSizeFromSequenceLength(sequence_length)
        residual_conn = []
        embedding = self.binary_embedding(input_logits)
        for i in range(
            self.ZERO, len(self.abstractionLayers), self.TWO
        ):  # Step through every other layer
            lookup_index = i // self.TWO
            if input_pad_size[lookup_index] == self.ONE:  # either 1 or 0
                embedding = self.padEmbedding(embedding, pad_direction)
            embedding = self.sparseTransformerForward(
                embedding, self.abstractionLayers[i], transformerArguments
            )
            residual_conn.append(embedding)
            embedding = self.abstractionForward(
                embedding, self.abstractionLayers[i + self.ONE]
            )
        # basically: n -> 2*n - mod
        if input_pad_size[-1] == self.ONE:
            embedding = self.padEmbedding(embedding, pad_direction)
        embedding = self.sparseTransformerForward(
            embedding, self.mainTransformer, transformerArguments
        )
        if input_pad_size[-1] == self.ONE:
            embedding = self.chopEmbedding(embedding, pad_direction)
        for i in range(
            self.ZERO, len(self.deabstractionLayers), self.TWO
        ):  # Step through every other layer
            lookup_index = self.abstraction_level - i // self.TWO - self.TWO
            embedding = self.deabstractionForward(
                embedding, self.deabstractionLayers[i + self.ONE]
            )
            embedding += residual_conn[lookup_index]
            embedding = self.sparseTransformerForward(
                embedding, self.deabstractionLayers[i], transformerArguments
            )
            if input_pad_size[lookup_index] == self.ONE:
                embedding = self.chopEmbedding(embedding, pad_direction)

        output_logits = self.decode_embedding(embedding)
        return output_logits


# myTransformer = HierachicalTokenizationTransformer()
# myTransformer = HierachicalTokenizationTransformer(abstraction_level=5, num_layers=10)
# myTransformer = HierachicalTokenizationTransformer(abstraction_level=100, num_layers=2)
myTransformer = HierachicalTokenizationTransformer(abstraction_level=20, num_layers=2)
# input_data = torch.ones(20, 1000, dtype=torch.long)  # batch size: 20, sequence length: 30
input_data = torch.ones(20, 30, dtype=torch.long)  # batch size: 20, sequence length: 30
output_data = myTransformer.forward(
    input_data,
    pad_direction="right",
    transformerArguments=TransformerArguments(is_causal=True),
)
print(output_data.shape)  # 20, 30, 2
