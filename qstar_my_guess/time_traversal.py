import torch

# q+astar & mcts

# heuristic: compute cosine similarity between target token and actual token
# sequence heuristic: seqlen heuristic

# where do targets come from?
# historical tokens: reverse order autoregressive model predictions, memory retrieval, past context
# future tokens: future predictions that being made short or slow (skipping intermediate steps, or making it faster), or contradict with common future predictions (unusual, less probable future tokens)

# TODO: hierarchy of control (by two special tokens: <abstract> and <deabstract>. one can set max abstraction level)
# TODO: slowing down and speeding up
output = speed_adjustment()
# TODO: emitting multiple kinds of tokens at the same time, at separate channels
# TODO: rollback gradient descent when no further improvement is found

# v1: separated world model & control model

init_sequence = world_model(noise)
action_sequence = action_model(init_sequence, prompt)
continue_sequence = world_model(init_sequence + action_sequence)
similarity = torch.cosine_similarity(continue_sequence, prompt)

# v2: unified word model & control model
init_sequence = uniform_model(
    noise, mode=world
)  # unified model will need some token generation restriction.
action_sequence = uniform_model(init_sequence + prompt, mode=action)
continue_sequence = uniform_model(init_sequence, action_sequence, mode=world)
similarity = torch.cosine_similarity(continue_sequence, prompt)

# v3: real world evaluators, could be no time traversal, but can change previous prompt (regret, backpropagate, (optionally) forget (maximize gradient) old (wrong) prediction and learn (minimize gradient) actual (real) prediction)

init_sequence = real_world(
    random_actions
)  # does this real world support time traversal?

if real_world.traversable:  # @property
    node_hash = (
        real_world.commit()
    )  # usually this is automatically handled? no need to intentionally commit?
    if not traverse_complete:
        real_world.rollback(node_hash)
else:
    prompter_remember(current_prompt, current_outcome)
    actor_remember(current_action, current_outcome)
    actor_regret(prompt, current_action, target)
    prompter_regret(prompt, target)# will change the prompt manufacturer
    # prompt = prompt_manufacturer(target) -> action -> current_outcome
    # delta_prompt, delta_action, delta_current_outcome -> closer than target

    # you may add some noise when mismatch found.

# prompt shall be crafted on both input tokens and target tokens.

# use special token + target as prompt
prompt = special_token + target

# use thought token as prompt
prompt = get_thought_tokens(
    special_token + target, seqlen
)  # thought tokens has to be "understood" by the model, so that we can know its intention (can we convert natural language prompt to thought tokens, aligned?)
meaning = get_thought_meaning(prompt, token_space = english) # kind of like the machine using input method.

# perform gradient descent based on cosine similarity, more similar result to target, more learning.
# only perform descent on the most similar one, or the most plausible way.
# next time learn about the same pattern, we still select the most similar one, closing the gap.

# a*: minimize distance from target to result + distance from source (init state) to result

# the reward can also be related to "easyness". if the model is doing some easy stuff (predicted by the world model), then no intensive learning is performed. if what the model want is hard to realize in world, the model will learn hard to do it.
reward = calculate_reward(computational_time, loss_delta)

# q function shall be used with caution. it will have some weight against actual evaulation. only if it is trusted, we can give it some high weight value in order to reduce computation.