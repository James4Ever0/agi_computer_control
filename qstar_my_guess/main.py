# a-star is backtracking.
# monte carlo tree search during training?
# but how does alphazero works?

# a -> b -> c
#    \ b1 -> c1
#    \ b2 -> c2

# i suspect that mcts can be done after ppo training.
# maybe it can be batched, but it must be slower.

# like the machine have multiple paths, but it can always return to previous state.
# for conversation we do not have such state to maintain yet, so does our action tokens.
# we can only evaluate the reward afterwards.
# however, if we pretend that we have the reward along the way we generate, maybe we can backpropagate and return the best state.

# it is unclear whether we have the reward function given explicitly or implicitly. too many things to reward. we only know the current state and the model needs to train a reward function by itself.

# so i suppose this model will retract its actions dynamically, based on value functions.
# once this model learns how to undone its generated tokens, consciousness comes up.

# and the reward function, is simply learned along the way, learned autoregressively. set the baseline as 0.5 and adjustable as -0.5 to 0.5, or baseline as 0 and adjustable as -1 to 1

# so i would give the model some "navigation" tokens like deletion, move left, move right and so on to manipulate ongoing sequences. these are synthetic data that are invariant to the representation system but are quite different to the ai model. i will give the model the right to pause, so that the info feeded in will not change, only the hidden state will. how to express that hidden state in the context of transformers?

# invariant transformations can be simplified to its simple flattened form, but can be augmented during training.

# the HID action space is somehow having some semantically invariant transformation that is just unclear or too general, but it does have, and you can augment it however you want, with no promise that it will result into the same outcome.

# to remind you further, you do not need anything alien to do media content automation. these things are spectacularly hard, especially in the context of thin air rather than media manipulation.

# you really are talented. you are wasting time just because you don't practice it in the right place.

# so you can treat the video generation as the same process of text generation, and use mcts to improve it.

# the video is abstract. you may generate high level features all the way down to segments and details.

################### how to develop hidden latent space ###################

# train multiple agents to watch video with random internal activities, average them out with others and find the tendency

