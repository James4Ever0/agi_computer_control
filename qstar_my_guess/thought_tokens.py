import torch
import math
from beartype import beartype
import torch.nn.functional as F
from strenum import StrEnum
from enum import auto, Enum
import copy

class InsertionMethodCategory(Enum):
    common_source = auto()
    separate_source = auto()

class ThoughtTokenInsertionMethod(Enum):
    autoregressive = (auto(), InsertionMethodCategory.common_source)
    generative_insert = (auto(), InsertionMethodCategory.common_source)
    iterate_and_insert_separately = (auto(), InsertionMethodCategory.separate_source)
    iterate_and_insert_together = (auto(), InsertionMethodCategory.separate_source)

    @property
    def category(self):
        return self.value[1]

def equality_fulfillment_transformer(instance):
    new_instance = copy.copy(instance)
    assert hasattr(instance, 'fulfilled'), "cannot process instance with 'fulfilled' attribute"
    setattr(new_instance, 'fulfilled', False)
    old_eq = copy.copy(new_instance.__eq__)
    def new_eq(self, other:object):
        is_equal = old_eq(other)
        if is_equal:
            self.fulfilled = True
        return is_equal
    setattr(new_instance, '__eq__', new_eq)
    return new_instance


class UnknownThoughtTokenInsertionMethod(Exception):
    def __init__(self, insert_method):
        super().__init__(f"Method '{insert_method}' is not available.")

@beartype
def get_batch_and_seqlen(source_tokens: torch.Tensor):
    source_size = source_tokens.shape
    assert (
        len(source_size) == 2
    ), f"wrong token size: {source_size} required: (batch, seqlen)"
    batch, seqlen = source_size
    return batch, seqlen


@beartype
def create_zeros_from_shape_and_insert_rate(
    batch: int, seqlen: int, insert_rate: float
):
    assert insert_rate > 0, f"insert rate not positive: {insert_rate}"
    added_seqlen = math.ceil(thought_token_insert_rate * seqlen)
    new_seqlen = seqlen + added_seqlen
    zeros = torch.ones((batch, new_seqlen))
    return added_seqlen, new_seqlen, zeros


@beartype
def create_mask(batch: int, seqlen: int, k: int):
    assert k > 0, f"k ({k}) is not positive"
    assert k < seqlen, f"k ({k}) must be less than seqlen ({seqlen})"

    # Generate random indices for each row
    random_indices = torch.stack([torch.randperm(seqlen)[:k] for _ in range(batch)])

    # Create a mask tensor to mark the selected indices
    mask = torch.zeros((batch, seqlen), dtype=torch.bool)
    mask.scatter_(1, random_indices, True)

    return mask


@beartype
def insert_source_token_to_zeros(
    source_tokens: torch.Tensor,
    zeros: torch.Tensor,
    batch: int,
    seqlen: int,
    new_seqlen: int,
):
    source_token_locations = create_mask(batch, new_seqlen, seqlen)
    zeros[source_token_locations] = source_tokens
    return source_token_locations


@beartype
def insert_thought_token_to_zeros(
    thought_tokens: torch.Tensor,
    zeros: torch.Tensor,
    source_token_locations: torch.Tensor,
):
    thought_token_locations = not source_token_locations
    zeros[thought_token_locations] = thought_tokens
    return thought_token_locations


@beartype
def sample_thought_tokens(
    thought_token_vocabulary: list[int], batch: int, added_seqlen: int
):
    # Sampled thought_token_vocabulary indices
    sampled_indices = torch.randint(
        0, len(thought_token_vocabulary), size=(batch, added_seqlen)
    )

    # Create tensor using sampled indices
    thought_tokens = torch.tensor(thought_token_vocabulary)[sampled_indices]

    return thought_tokens


@beartype
def insert_thought_tokens(
    source_tokens: torch.Tensor,
    thought_token_vocabulary: list[int],
    thought_token_insert_rate: float,
):
    batch, seqlen = get_batch_and_seqlen(source_tokens)
    added_seqlen, new_seqlen, zeros = create_zeros_from_shape_and_insert_rate(
        batch, seqlen, thought_token_insert_rate
    )
    source_token_locations = insert_source_token_to_zeros(
        source_tokens, zeros, batch, seqlen, new_seqlen
    )
    thought_tokens = sample_thought_tokens(
        thought_token_vocabulary, batch, added_seqlen
    )
    thought_token_locations = insert_thought_token_to_zeros(thought_tokens, zeros, source_token_locations)
    return zeros, new_seqlen, source_token_locations, thought_token_locations


@beartype
def pad_seq_left(input_tensor:torch.Tensor, pad_size:int, value):
    assert pad_size >= 0, f"pad size ({pad_size}) must be non negative"
    ret = F.pad(input_tensor, (pad_size, 0), mode='constant', value=value)
    return ret

@beartype
def insert_thought_tokens_and_yield_train_pairs(source_tokens:torch.Tensor, thought_token_vocabulary:list[int], thought_token_insert_rate:float, insertion_method: ThoughtTokenInsertionMethod):
    insertion_method = equality_fulfillment_transformer(insertion_method)
    if insertion_method.category == InsertionMethodCategory.common_source:
        (
            processed_tokens,
            new_seqlen,
            source_token_locations,
            thought_token_locations,
        ) = insert_thought_tokens(
            source_tokens, thought_token_vocabulary, thought_token_insert_rate
        )
        if insertion_method == ThoughtTokenInsertionMethod.autoregressive:
            yield from autoregressively_yield_train_pairs(processed_tokens, train_window_size, pad_token_idx,new_seqlen)
        
        elif insertion_method == ThoughtTokenInsertionMethod.generative_insert:
            yield from ...
    else:
        if insertion_method == ThoughtTokenInsertionMethod.iterate_and_insert_together:
            yield from ...
        
        elif insertion_method == ThoughtTokenInsertionMethod.iterate_and_insert_separately:
            yield from ...
    if not insertion_method.fulfilled:
        raise UnknownThoughtTokenInsertionMethod(insertion_method)

# the sample process shall start from zero.
def autoregressively_yield_train_pairs(processed_tokens:torch.Tensor, train_window_size:int, pad_token_idx:int,new_seqlen:int):
    assert new_seqlen > 1
    padded_processed_tokens = pad_seq_left(processed_tokens, train_window_size - 1, pad_token_idx)

    for i in range(new_seqlen - 1):
        input_tokens = padded_processed_tokens[:, i:i+train_window_size]
        target_tokens = padded_processed_tokens[:, i+1:i+train_window_size+1]
        yield input_tokens, target_tokens

### another version:

# v1.1
for ... in ...:
    input_tokens = insert_thought_tokens(padded_source_tokens[:, :-1]) # fully randomed input tokens.
    target_tokens = insert_thought_tokens(padded_source_tokens[:, 1:])

# v1.2

for ... in ...:
    insert_thought_tokens(padded_source_tokens[:, :])
    for ... in ...:
        yield ..., ...

# v2
for ... in ...:
    input_tokens = processed_tokens[1:] # still read the original processed tokens,
    output_token_prob = language_model(input_tokens)
    target_token_mask = ...
    target_tokens = generate_target_tokens(output_token_prob, target_token_mask)
    yield input_tokens, target_tokens

# output thought tokens affect input tokens?

if __name__ == '__main__':
    # begin test

    base_token_count = 1000
    thought_token_count = 2000
    pad_token_idx = base_token_count + thought_token_count
    total_token_count = pad_token_idx + 1

    thought_token_insert_rate = 0.2
    source_batchsize = 1
    source_seqlen = 20
    source_size = (source_batchsize, source_seqlen)
    train_window_size = 10

    thought_token_vocabulary = [base_token_count + i for i in range(thought_token_count)]

    source_tokens = torch.randint(
        0, base_token_count, source_size
    )  # okay, lower than upper bound.

    for _ in insert_thought_tokens_and_yield_train_pairs(source_tokens, thought_token_vocabulary, thought_token_insert_rate, ThoughtTokenInsertionMethod.autoregressive): ...