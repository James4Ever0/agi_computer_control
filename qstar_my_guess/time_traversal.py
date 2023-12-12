import torch

# q+astar & mcts

# heuristic: compute cosine similarity between target token and actual token
# sequence heuristic: seqlen heuristic

# where do targets come from?
# historical tokens: reverse order autoregressive model predictions, memory retrieval, past context
# future tokens: future predictions that being made short or slow (skipping intermediate steps, or making it faster), or contradict with common future predictions (unusual, less probable future tokens)

# TODO: hierarchy of control

# v1: separated world model & control model
init_sequence = world_model(noise)

# v2: unified word model & control model
init_sequence = uniform_model(noise, mode=world) # unified model will need some token generation restriction.

# v3: real world evaluators, no time traversal, but can change previous prompt (regret, backpropagate, (optionally) forget (maximize gradient) old (wrong) prediction  and learn (minimize gradient) actual (real) prediction)
init_sequence = real_world(random_actions) # does this real world support time traversal?

if real_world.traversable: # @property
    node_hash = real_world.commit() # usually this is automatically handled? no need to intentionally commit?
    real_world.rollback(node_hash)

# use special token + target as prompt
prompt = special_token + target

# use thought token as prompt
prompt = get_thought_tokens(special_token + target, seqlen) # thought tokens has to be "understood" by the model, so that we can know its intention (can we convert natural language prompt to thought tokens, aligned?)

# perform gradient descent based on cosine similarity, more similar result to target, more learning.
# only perform descent on the most similar one.
# next time learn about the same pattern, we still select the most similar one, closing the gap.

# a*: minimize distance from target to result + distance from source (init state) to result

# the reward can also be related to "easyness". if the model is doing some easy stuff (predicted by the world model), then no intensive learning is performed. if what the model want is hard to realize in world, the model will learn hard to do it.