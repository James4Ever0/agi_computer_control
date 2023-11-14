import torch

# let's use networkx to generate random graphs?

# the method to train it? we only keep the input & output the same as the static model.

# place some clock like neurons, just like the SNN
# to ensure that neurons won't get far away from each other
# if they want to have some spikes

# now you might want to use solver for network assembly.

# neurons that fire together, wire together

# there are pending connections. if these connections has larger gradients, they will be welcomed.

# calculate route to output. if there is no route to output, then no gradient.

# it rolls out the monte carlo tree search in ppo. would you find some optimal neural network structure with that?

class Connection:
    def __init__(self):
        self.connected_index = ...
        self.weights = ...
        self.bias = ...

class Neuron:
    def __init__(self): # you can choose like: 100x1 -> 1x50 (1 can be larger, like 2) or 10x10
        # but 100x1, 1x50 can be perferred, since it includes the potential or hidden state
        # we can also tune the decay factor inside the neuron.
        self.index = ...
        self.current_potential = ...
        self.input_connection = ...
        self.output_connection = ...

# so we would iterate through our connection pool, propagate
# how to do batch training?

# input_neurons = ...
# output_neurons = ...

# calculate inter_neuron distance
# we have to use sparse matrix

# output_node_index, input_node_index
# if you swap, you break the connection
# you can compute the gradient, to decide if you want to break or keep
# (2, 3)
# (4, 1)

# we will start with the input neurons, iterate over the network

# it forms a loop.

# output: spikes, movements per neuron

# what determines the connectivity between neurons?
# what forms new connectivities?

# (in0, out0)
# (in1, out1)
# (in2, out2)

# i think we do not need the coordinates.
# i think we need some gcn, or recommendation engine?

# just represent the connectivity in sparse matrix.

# connection swapping?

# minimize the number of unconnected neurons
# cannot connect to themselves
# you may have pre-programmed randomly initialized connectivity matrix

# hierarchical grouping?

# would you rather do some value swapping in weight matricies

# liquid state machine

# dynamic rewiring of neurons by upper/lower weight limit, or the free energy principle

# or you can also look at the actual energy consumption or battery consumption

# by far multiple thesis have been stated, but yet unproven and unimplemented.

# the primary objective of this project is to create computer operating bots that can understand human intentions and do everyday tasks on their own, including browsing, coding and searching.

# it is not clear whether this project will create some artificial life, and it is not our objective. it might be a good research project for the bots, but i can say for sure it is not for me. i have limited capacity of knowledge and resources. i don't allocate that much memory biologically or physically. i choose to build the bot first. it does not have to be that complex that i cannot imagine or create.

# decision matters. the executors of this 'life' project should be bots. i am the project manager. besides, i could participate in the 'cybergod' project which controls the computer by computer itself.

# noise in, words out, recurse. that is dream.
