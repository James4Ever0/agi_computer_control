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


class MultiheadAttentionStack(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, **kwargs):
        super(MultiheadAttentionStack, self).__init__()
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=embed_dim, num_heads=num_heads, **kwargs
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        input_tensor: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        output = input_tensor
        for layer in self.layers:
            output, _ = layer(
                query=output,
                key=output,
                value=output,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
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

        for _ in range(abstraction_level):
            # Create the attention and abstraction layers
            att_layer = MultiheadAttentionStack(
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
            datt_layer = MultiheadAttentionStack(
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

    def forward(
        self,
        input_logits: Tensor,
        pad_direction: Literal["left", "right"],
        key_padding_mask: Tensor | None = None,
        need_weights: bool = True,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):  # it is trimmed from one side. is it causal?
        assert (
            len(input_logits.shape) == self.TWO
        ), "input logits shall be of shape (batch_size, sequence_length)"
        _, sequence_length = input_logits.shape
        assert sequence_length != self.ZERO, "zero length sequence encountered"
        # div, mod = divmod(sequence_length, self.pad_size)
        # pad_size = 0 if mod == 0 else self.pad_size-mod
        # pad_input_logits = F.pad(input_logits, (0, pad_size, 0, 0), "constant", 0)
        # print(pad_input_logits.shape)
        input_pad_size = []
        residual_conn = []
        msequence_length = int(sequence_length)
        for _ in range(self.abstraction_level):
            div, mod = divmod(msequence_length, self.TWO)
            if mod == self.ONE:
                input_pad_size.append(self.ONE)
                div += self.ONE
            else:
                input_pad_size.append(self.ZERO)
            msequence_length = div
        # breakpoint()
        embedding = self.binary_embedding(input_logits)
        # embedding = self.binary_embedding(pad_input_logits)
        for i in range(
            self.ZERO, len(self.abstractionLayers), self.TWO
        ):  # Step through every other layer
            embedding = self.abstractionLayers[i](
                embedding,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )  # 20, 30, 768
            # make sure it is divisible by 2
            residual_conn.append(embedding)
            if input_pad_size[i // self.TWO] == self.ONE:  # either 1 or 0
                # print('padding')
                embedding = F.pad(
                    embedding,
                    (self.ZERO, self.ZERO, self.ZERO, self.ONE)
                    if pad_direction == "right"
                    else (self.ZERO, self.ZERO, self.ONE, self.ZERO),
                    "constant",
                    self.ZERO,
                )
            # print(embedding.shape)
            # breakpoint()
            embedding = einops.rearrange(
                embedding, "b (s1 g) d -> b d s1 g", g=self.TWO
            )
            embedding = self.abstractionLayers[i + self.ONE](
                embedding
            )  # Apply attention and abstraction
            embedding = einops.rearrange(embedding, f"b d s {self.ONE} -> b s d")
        # basically: n -> 2*n - mod
        for i in range(
            self.ZERO, len(self.deabstractionLayers), self.TWO
        ):  # Step through every other layer
            embedding = self.deabstractionLayers[i](
                embedding,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
            embedding = einops.rearrange(embedding, f"b s d -> b d s {self.ONE}")
            embedding = self.deabstractionLayers[i + self.ONE](
                embedding
            )  # Apply inverse attention and deabstraction
            embedding = einops.rearrange(
                embedding, "b d s1 g -> b (s1 g) d", g=self.TWO
            )
            if (
                input_pad_size[self.abstraction_level - i // self.TWO - self.ONE]
                == self.ONE
            ):
                embedding = (
                    embedding[:, : -self.ONE, :]
                    if pad_direction == "right"
                    else embedding[:, self.ONE :, :]
                )
            embedding += residual_conn[self.abstraction_level - i // self.TWO - self.ONE]

        output_logits = self.decode_embedding(embedding)
        # pad_output_logits = self.decode_embedding(embedding)
        # output_logits = pad_output_logits[:, :sequence_length]
        return output_logits


myTransformer = HierachicalTokenizationTransformer()
# myTransformer = HierachicalTokenizationTransformer(abstraction_level=5, num_layers=10)
# myTransformer = HierachicalTokenizationTransformer(abstraction_level=20, num_layers=2)
# input_data = torch.ones(20, 1000, dtype=torch.long)  # batch size: 20, sequence length: 30
input_data = torch.ones(20, 30, dtype=torch.long)  # batch size: 20, sequence length: 30
output_data = myTransformer.forward(input_data, pad_direction="right", is_causal=True)
print(output_data.shape)  # 20, 30, 2
