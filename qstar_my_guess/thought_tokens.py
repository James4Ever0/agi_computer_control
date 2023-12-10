import torch

thought_token_size = 2000
thought_token_insert_rate = 0.2

source_tokens = ...
processed_tokens = insert_thought_tokens(source_tokens, thought_token_vocabulary, thought_token_insert_rate)

input_tokens = processed_tokens[:-1]
target_tokens = processed_tokens[1:]

### another version:

# v1
input_tokens = insert_thought_tokens(...) # fully randomed input tokens.
# v2
input_tokens = processed_tokens[1:]
output_token_prob = ...
target_tokens = generate_target_tokens(output_token_prob, target_token_mask)

# output thought tokens affect input tokens?