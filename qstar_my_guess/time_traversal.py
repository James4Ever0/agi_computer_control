import torch

# heuristic: compute cosine similarity between target token and actual token

# v1: separated word model & control model
init_sequence = world_model(noise)

# v2: unified word model & control model
init_sequence = uniform_model(noise)

# v3: real world evaluators, no time traversal, but can change previous prompt (regret, backpropagate, (optionally) forget (maximize gradient) old (wrong) prediction  and learn (minimize gradient) actual (real) prediction)
init_sequence = real_world()

# use special token + target as prompt
prompt = special_token + target

# use thought token as prompt
prompt = random_thought_tokens

# perform gradient descent based on cosine similarity, more similar result to target, more learning.
# only perform descent on the most similar one.
# next time learn about the same pattern, we still select the most similar one, closing the gap.

# the reward can also be related to "easyness". if the model is doing some easy stuff (predicted by the world model), then no intensive learning is performed. if what the model want is hard to realize in world, the model will learn hard to do it.