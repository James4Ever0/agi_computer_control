import torch
import math
from beartype import beartype


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
def create_mask(batch:int, seqlen:int, k:int):
    assert k > 0
    assert k < seqlen
    
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
    thought_tokens = torch.Tensor()
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
    insert_thought_token_to_zeros(thought_tokens, zeros, source_token_locations)
    processed_tokens = zeros
    return processed_tokens, source_token_locations, thought_token_locations


base_token_count = 1000
thought_token_count = 2000
total_token_count = base_token_count + thought_token_count

thought_token_insert_rate = 0.2
sample_size = (1, 20)


source_tokens = torch.randint(
    0, base_token_count, sample_size
)  # okay, lower than upper bound.
(
    processed_tokens,
    source_token_locations,
    thought_token_locations,
) = insert_thought_tokens(
    source_tokens, thought_token_vocabulary, thought_token_insert_rate
)

input_tokens = processed_tokens[:-1]
target_tokens = processed_tokens[1:]

### another version:

# v1
input_tokens = insert_thought_tokens(...)  # fully randomed input tokens.
# v2
input_tokens = processed_tokens[1:]
output_token_prob = ...
target_tokens = generate_target_tokens(output_token_prob, target_token_mask)

# output thought tokens affect input tokens?
