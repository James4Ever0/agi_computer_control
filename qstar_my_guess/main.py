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