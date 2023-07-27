# i wouldn't know if there is anything called "calculated type" in python "typing" module or erglang, but i do know some "type" details about numpy matrix multiplication and torch convolutional neural networks.

# Calculated types or custom type hint and type checkers

# There's been a long history of attempts on typechecking libraries like `pytorch` and `numpy`. One major issue is that passing different parameters to these constructors create different types (tensors with traits (matrix multiplication) and tensor functions). There's some relationship between definition of a convolutional network and what type of value (usually not a fixed type, but rather a set of types (infinite)) it can accept and return.

# So I propose or want to know how we can let Erg to solve this long-standing problem, by letting the static typechecker to know the type of `torch.nn.Conv2d` and emit errors before execution? Help can be found by:

# ```python
# import torch
# help(torch.nn.Conv2d)
# ```