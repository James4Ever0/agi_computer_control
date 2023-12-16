from typing import Callable, Iterable, Optional
import torch
import math
from beartype import beartype
from beartype.vale import Is
import torch.nn.functional as F
from enum import auto, Enum
import copy
from typing_extensions import overload, Literal, Annotated
from overtake import overtake  # type: ignore

ReplaceRatio = Annotated[float, Is[lambda number: 0 <= number < 1]]
NonNegativeFloat = Annotated[float, Is[lambda number: number > 0]]

class InsertionMethodCategory(Enum):
    common_source = auto()
    separate_source = auto()

class ThoughtTokenInsertionMethod(Enum):
    autoregressive = (auto(), InsertionMethodCategory.common_source)
    generative_insert = (auto(), InsertionMethodCategory.common_source)
    generative_insert_and_overwrite = ( # with probablistic noise and original random token swap ratio
        auto(),
        InsertionMethodCategory.common_source,
    )  # will use generated target tokens to replace original randomly inserted thought tokens.

    # not implemented
    # iterate_and_insert_separately = (auto(), InsertionMethodCategory.separate_source)
    # iterate_and_insert_together = (auto(), InsertionMethodCategory.separate_source)

    @property
    def category(self):
        return self.value[1]


def equality_fulfillment_instance_transformer(instance):
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
def create_zeros_from_tensor_metadata_and_insert_rate(
    batch: int, seqlen: int, dtype: torch.dtype,insert_rate: float
):
    assert insert_rate > 0, f"insert rate not positive: {insert_rate}"
    added_seqlen = math.ceil(thought_token_insert_rate * seqlen)
    new_seqlen = seqlen + added_seqlen
    zeros = torch.zeros((batch, new_seqlen), dtype=dtype)
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
    thought_token_insert_rate: NonNegativeFloat,
):
    batch, seqlen = get_batch_and_seqlen(source_tokens)
    added_seqlen, new_seqlen, zeros = create_zeros_from_tensor_metadata_and_insert_rate(
        batch, seqlen, source_tokens.dtype, thought_token_insert_rate
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
    thought_token_insert_rate: NonNegativeFloat,
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
    thought_token_insert_rate: NonNegativeFloat,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    (
        autoregressive_generator,
        _,
    ) = get_autoregressive_generator_and_thought_token_locations(
        source_tokens, thought_token_vocabulary, thought_token_insert_rate
    )
    yield from autoregressive_generator


def generative_insert_thought_tokens_and_yield_train_pairs(
    source_tokens: torch.Tensor,
    thought_token_vocabulary: list[int],
    thought_token_insert_rate: NonNegativeFloat,
    non_thought_token_vocabulary: list[int],
    target_token_prob_generator: Callable[[torch.Tensor], torch.Tensor],
    probablistic_noise_ratio:ReplaceRatio = 0,
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
        probablistic_noise_ratio
    )


@overload
def insert_thought_tokens_and_yield_train_pairs(
    insertion_method: Literal[ThoughtTokenInsertionMethod.generative_insert],
    source_tokens: torch.Tensor,
    thought_token_vocabulary: list[int],
    thought_token_insert_rate: NonNegativeFloat,
    non_thought_token_vocabulary: list[int],
    target_token_prob_generator: Callable[[torch.Tensor], torch.Tensor],
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    yield from generative_insert_thought_tokens_and_yield_train_pairs(
        source_tokens,
        thought_token_vocabulary,
        thought_token_insert_rate,
        non_thought_token_vocabulary,
        target_token_prob_generator,
    )

@beartype
def generate_porportional_mask_for_tensor(input_tensor:torch.Tensor, porportion: ReplaceRatio):
    # Determine the number of elements to be zeroed
    num_elements = input_tensor.numel()
    num_zero_elements = int(porportion * num_elements)

    # Create a boolean mask for randomly selecting 30% of the elements
    mask = torch.zeros(num_elements, dtype=torch.bool)
    mask[:num_zero_elements] = 1  # Set the first 30% elements to True
    mask = mask[torch.randperm(num_elements)]  # Shuffle the mask randomly
    return mask

@beartype
def swap_input_tokens_with_previous_target_tokens_by_swap_ratio(input_tokens:torch.Tensor,prev_target_tokens:torch.Tensor, input_token_swap_ratio):
    input_mask = generate_porportional_mask_for_tensor(input_tokens, input_token_swap_ratio)
    prev_target_mask = ~input_mask
    input_tokens[input_mask] = 0
    prev_target_tokens[prev_target_mask] = 0
    input_tokens += prev_target_tokens
    return input_tokens

@overload
def insert_thought_tokens_and_yield_train_pairs(
    insertion_method: Literal[
        ThoughtTokenInsertionMethod.generative_insert_and_overwrite
    ],
    source_tokens: torch.Tensor,
    thought_token_vocabulary: list[int],
    thought_token_insert_rate: NonNegativeFloat,
    non_thought_token_vocabulary: list[int],
    target_token_prob_generator: torch.nn.Module,
    probablistic_noise_ratio:ReplaceRatio,
    input_token_swap_ratio:ReplaceRatio,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    generator = generative_insert_thought_tokens_and_yield_train_pairs(
        source_tokens,
        thought_token_vocabulary,
        thought_token_insert_rate,
        non_thought_token_vocabulary,
        target_token_prob_generator,
        probablistic_noise_ratio,
    )
    prev_target_tokens = None
    for input_tokens, target_tokens in generator:
        if prev_target_tokens is not None:
            if input_token_swap_ratio > 0:
                input_tokens = swap_input_tokens_with_previous_target_tokens_by_swap_ratio(input_tokens, prev_target_tokens, input_token_swap_ratio)
            else:
                input_tokens = prev_target_tokens

        yield input_tokens, target_tokens
        prev_target_tokens = target_tokens.clone()


@overtake(runtime_type_checker="beartype")
def insert_thought_tokens_and_yield_train_pairs(
    insertion_method,
    source_tokens,
    thought_token_vocabulary,
    thought_token_insert_rate,
    non_thought_token_vocabulary=None,
    target_token_prob_generator=None,
    probablistic_noise_ratio = 0,
    input_token_swap_ratio = 0
):
    ...


@beartype
def crop_input_token_by_index_and_window_size(
    processed_tokens: torch.Tensor, index: int, window_size: int
):
    cropped_tokens = processed_tokens[:, index : index + window_size]
    return cropped_tokens


@beartype
def crop_target_token_by_index_and_window_size(
    processed_tokens: torch.Tensor, index: int, window_size: int
):
    return crop_input_token_by_index_and_window_size(
        processed_tokens, index + 1, window_size
    )


# the sample process shall start from zero.
@beartype
def autoregressively_yield_train_pairs(
    padded_processed_tokens: torch.Tensor, train_window_size: int, new_seqlen: int
):
    for i in range(new_seqlen - 1):
        input_tokens = crop_input_token_by_index_and_window_size(
            padded_processed_tokens, i, train_window_size
        )
        target_tokens = crop_target_token_by_index_and_window_size(
            padded_processed_tokens, i, train_window_size
        )
        yield input_tokens, target_tokens


@beartype
def prob_to_token(
    token_prob: torch.Tensor,
    masked_location: Optional[torch.Tensor] = None,
    masked_vocabulary: Optional[list[int]] = None,
):
    ret_prob = token_prob.clone()
    if masked_vocabulary is not None:
        ret_prob[:, masked_vocabulary] = 0
    ret_tokens = torch.argmax(ret_prob, dim=2)
    if masked_location is not None:
        ret_tokens[masked_location] = 0
    return ret_tokens


@beartype
def generate_target_tokens_with_thought_token_loctions_and_non_thought_token_vocabulary(
    token_prob: torch.Tensor,
    thought_token_locations: torch.Tensor,
    thought_token_vocabulary: list[int],
    non_thought_token_vocabulary: list[int],
):
    assert (
        len(token_prob.shape) == 3
    ), f"wrong token probability tensor shape ({token_prob}). should be: (batch_size, sequence_length, vocabulary_size)"
    # what is the shape of this prob?
    non_thought_token_locations = ~thought_token_locations

    thought_tokens = prob_to_token(
        token_prob, non_thought_token_locations, non_thought_token_vocabulary
    )
    non_thought_tokens = prob_to_token(
        token_prob, thought_token_locations, thought_token_vocabulary
    )
    ret_tokens = thought_tokens + non_thought_tokens

    return ret_tokens

@beartype
def generate_gaussian_noise_within_bounds(size:tuple, lower:float, upper:float):
    assert lower <= upper, f"rule lower ({lower}) <= upper ({upper}) does not comply"
    # Parameters
    mean = 0
    std = 1
    # Generate Gaussian noise
    noise = torch.normal(mean, std, size=size)  # Generate 10 samples of Gaussian noise

    # Scale the noise to the range [a, b]
    scaled_noise = (noise - noise.mean()) / noise.std()  # Standardize the noise
    scaled_noise = (scaled_noise * (upper - lower)) + (lower + upper) / 2  # Scale to the desired range
    return scaled_noise


@beartype
def add_probablistic_noise_to_prob(token_prob:torch.Tensor, probablistic_noise_ratio:ReplaceRatio):
    min_prob = float(token_prob.min())
    max_prob = float(token_prob.max())
    noise_prob = generate_gaussian_noise_within_bounds(token_prob.shape, min_prob, max_prob)
    token_prob_with_noise = token_prob + noise_prob * probablistic_noise_ratio
    return token_prob_with_noise

# demo on how to use thought tokens.
@beartype
def generative_insert_yield_train_pairs(
    autoregressive_generator: Iterable,
    target_token_prob_generator: Callable[[torch.Tensor], torch.Tensor],
    padded_thought_token_locations: torch.Tensor,
    thought_token_vocabulary: list[int],
    non_thought_token_vocabulary: list[int],
    train_window_size: int,
    probablistic_noise_ratio: ReplaceRatio,
):
    for i, (input_tokens, _) in enumerate(autoregressive_generator):
        with torch.no_grad():
            output_token_prob = target_token_prob_generator(input_tokens)
            output_token_prob = add_probablistic_noise_to_prob(output_token_prob,probablistic_noise_ratio)
        thought_token_locations = crop_target_token_by_index_and_window_size(
            padded_thought_token_locations, i, train_window_size
        )
        target_tokens = generate_target_tokens_with_thought_token_loctions_and_non_thought_token_vocabulary(
            output_token_prob,
            thought_token_locations,
            thought_token_vocabulary,
            non_thought_token_vocabulary,
        )
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
    non_thought_token_vocabulary = [
        i for i in range(total_token_count) if i not in thought_token_vocabulary
    ]

    source_tokens = torch.randint(
        0, base_token_count, source_size
    )  # okay, lower than upper bound.

    print("[autoregressive]".center(50, "-"))
    for input_tokens, target_tokens in insert_thought_tokens_and_yield_train_pairs(
        ThoughtTokenInsertionMethod.autoregressive,
        source_tokens,
        thought_token_vocabulary,
        thought_token_insert_rate,
    ):  
        print(input_tokens)
        print(target_tokens)
        print("-"*50)
        # breakpoint()
