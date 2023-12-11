from re import T
from typing import Callable, Iterable
import torch
import math
from beartype import beartype
import torch.nn.functional as F
from enum import auto, Enum
import copy
from typing_extensions import overload, Literal
from overtake import overtake  # type: ignore


class InsertionMethodCategory(Enum):
    common_source = auto()
    separate_source = auto()


class ThoughtTokenInsertionMethod(Enum):
    autoregressive = (auto(), InsertionMethodCategory.common_source)
    generative_insert = (auto(), InsertionMethodCategory.common_source)
    # not implemented
    iterate_and_insert_separately = (auto(), InsertionMethodCategory.separate_source)
    iterate_and_insert_together = (auto(), InsertionMethodCategory.separate_source)

    @property
    def category(self):
        return self.value[1]


def equality_fulfillment_transformer(instance):
    new_instance = copy.copy(instance)
    assert hasattr(
        instance, "fulfilled"
    ), "cannot process instance with 'fulfilled' attribute"
    setattr(new_instance, "fulfilled", False)
    old_eq = copy.copy(new_instance.__eq__)

    def new_eq(self, other: object):
        is_equal = old_eq(other)
        if is_equal:
            self.fulfilled = True
        return is_equal

    setattr(new_instance, "__eq__", new_eq)
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
    thought_token_locations = ~source_token_locations  # do not use "not"
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
    thought_token_locations = insert_thought_token_to_zeros(
        thought_tokens, zeros, source_token_locations
    )
    return zeros, new_seqlen, source_token_locations, thought_token_locations


@beartype
def pad_seq_left(input_tensor: torch.Tensor, pad_size: int, value):
    assert pad_size >= 0, f"pad size ({pad_size}) must be non negative"
    ret = F.pad(input_tensor, (pad_size, 0), mode="constant", value=value)
    return ret


@beartype
def pad_processed_and_thought_tokens(
    processed_tokens: torch.Tensor,
    thought_token_locations: torch.Tensor,
    train_window_size: int,
    pad_token_idx: int,
):
    pad_size = train_window_size - 1
    padded_processed_tokens = pad_seq_left(processed_tokens, pad_size, pad_token_idx)
    padded_thought_token_locations = pad_seq_left(
        thought_token_locations, pad_size, False
    )
    return padded_processed_tokens, padded_thought_token_locations


@beartype
def get_autoregressive_generator_and_thought_token_locations(
    source_tokens: torch.Tensor,
    thought_token_vocabulary: list[int],
    thought_token_insert_rate: float,
):
    (
        processed_tokens,
        new_seqlen,
        _,
        thought_token_locations,
    ) = insert_thought_tokens(
        source_tokens, thought_token_vocabulary, thought_token_insert_rate
    )

    assert new_seqlen > 1
    (
        padded_processed_tokens,
        padded_thought_token_locations,
    ) = pad_processed_and_thought_tokens(
        processed_tokens, thought_token_locations, train_window_size, pad_token_idx
    )
    autoregressive_generator = autoregressively_yield_train_pairs(
        padded_processed_tokens, train_window_size, new_seqlen
    )
    return autoregressive_generator, padded_thought_token_locations


@overload
def insert_thought_tokens_and_yield_train_pairs(
    insertion_method: Literal[ThoughtTokenInsertionMethod.autoregressive],
    source_tokens: torch.Tensor,
    thought_token_vocabulary: list[int],
    thought_token_insert_rate: float,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    (
        autoregressive_generator,
        _,
    ) = get_autoregressive_generator_and_thought_token_locations(
        source_tokens, thought_token_vocabulary, thought_token_insert_rate
    )
    yield from autoregressive_generator


@overload
def insert_thought_tokens_and_yield_train_pairs(
    insertion_method: Literal[ThoughtTokenInsertionMethod.generative_insert],
    source_tokens: torch.Tensor,
    thought_token_vocabulary: list[int],
    thought_token_insert_rate: float,
    non_thought_token_vocabulary: list[int],
    target_token_prob_generator: torch.nn.Module,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    (
        autoregressive_generator,
        padded_thought_token_locations,
    ) = get_autoregressive_generator_and_thought_token_locations(
        source_tokens, thought_token_vocabulary, thought_token_insert_rate
    )
    yield from generative_insert_yield_train_pairs(
        autoregressive_generator,
        target_token_prob_generator,
        padded_thought_token_locations,
        thought_token_vocabulary,
        non_thought_token_vocabulary,
        train_window_size,
    )


@overtake(runtime_type_checker="beartype")
def insert_thought_tokens_and_yield_train_pairs(
    insertion_method,
    source_tokens,
    thought_token_vocabulary,
    thought_token_insert_rate,
    non_thought_token_vocabulary=None,
    target_token_prob_generator=None,
):
    ...

@beartype
def crop_input_token_by_index_and_window_size(processed_tokens:torch.Tensor, index:int, window_size:int):
    cropped_tokens = processed_tokens[:, index : index + window_size]
    return cropped_tokens

@beartype
def crop_target_token_by_index_and_window_size(processed_tokens:torch.Tensor, index:int, window_size:int):
    return crop_input_token_by_index_and_window_size(processed_tokens, index+1, window_size)

# the sample process shall start from zero.
@beartype
def autoregressively_yield_train_pairs(
    padded_processed_tokens: torch.Tensor, train_window_size: int, new_seqlen: int
):
    for i in range(new_seqlen - 1):
        input_tokens = crop_input_token_by_index_and_window_size(padded_processed_tokens, i, train_window_size)
        target_tokens = crop_target_token_by_index_and_window_size(padded_processed_tokens, i, train_window_size)
        yield input_tokens, target_tokens

@beartype
def prob_to_token(token_prob: torch.Tensor, masked_location:torch.Tensor, masked_vocabulary:list[int]):
    ret_prob = token_prob.clone()
    ret_prob[masked_location] = 0
    ret_prob[:, masked_vocabulary] = 0
    ret_tokens = torch.argmax(ret_prob, dim = 2)
    ret_tokens[masked_location] = 0
    return ret_tokens


@beartype
def generate_target_tokens_with_thought_token_loctions_and_non_thought_token_vocabulary(token_prob: torch.Tensor, thought_token_locations: torch.Tensor, thought_token_vocabulary:list[int], non_thought_token_vocabulary: list[int]):
    assert len(token_prob.shape) == 3, f"wrong token probability tensor shape ({token_prob}). should be: (batch_size, sequence_length, vocabulary_size)"
    # what is the shape of this prob?
    non_thought_token_locations = ~thought_token_locations

    thought_tokens = prob_to_token(token_prob, non_thought_token_locations,non_thought_token_vocabulary)
    non_thought_tokens = prob_to_token(token_prob, thought_token_locations, thought_token_vocabulary)
    ret_tokens = thought_tokens + non_thought_tokens

    return ret_tokens

# demo on how to use thought tokens.
@beartype
def generative_insert_yield_train_pairs(
    autoregressive_generator: Iterable,
    target_token_prob_generator: Callable,
    padded_thought_token_locations: torch.Tensor,
    thought_token_vocabulary:list[int],
    non_thought_token_vocabulary: list[int],
    train_window_size: int,
):
    for i, (input_tokens, _) in enumerate(autoregressive_generator):
        with torch.no_grad():
            output_token_prob = target_token_prob_generator(input_tokens)
        thought_token_locations = crop_target_token_by_index_and_window_size(padded_thought_token_locations, i, train_window_size)
        target_tokens = generate_target_tokens_with_thought_token_loctions_and_non_thought_token_vocabulary(output_token_prob, thought_token_locations,thought_token_vocabulary, non_thought_token_vocabulary)
        yield input_tokens, target_tokens


# output thought tokens affect input tokens?

if __name__ == "__main__":
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

    thought_token_vocabulary = [
        base_token_count + i for i in range(thought_token_count)
    ]
    non_thought_token_vocabulary = [i for i in range(total_token_count) if i not in thought_token_vocabulary]

    source_tokens = torch.randint(
        0, base_token_count, source_size
    )  # okay, lower than upper bound.

    for _ in insert_thought_tokens_and_yield_train_pairs(
        ThoughtTokenInsertionMethod.autoregressive,
        source_tokens,
        thought_token_vocabulary,
        thought_token_insert_rate,
    ):
        ...
