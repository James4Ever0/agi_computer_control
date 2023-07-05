# link: https://taoa.io/posts/Shape-typing-numpy-with-pyright-and-variadic-generics

import numpy as np

from typing import Any
from typing_extensions import Unpack, TypeVarTuple
Shape = TypeVarTuple("Shape")

NDArray = np.ndarray[Unpack[Shape], np.dtype]
# from numpy.typing import NDArray

from typing import Tuple, TypeVar, Literal

# Generic dimension sizes types

T1 = TypeVar("T1", bound=int)

T2 = TypeVar("T2", bound=int)

T3 = TypeVar("T3", bound=int)

# Dimension types represented as typles



def rand_normal_matrix(shape: Tuple[Unpack[Shape]]) -> NDArray[Unpack[Shape], np.float64]:

    """Return a random ND normal matrix."""

    return np.random.standard_normal(size=shape)


# Yay correctly typed 2x2x2 cube!

LENGTH = Literal[2]

cube: NDArray[LENGTH, LENGTH, LENGTH, np.float64] = rand_normal_matrix((2,2,2))
myshape = (2,2,2) # tuple[Literal[2], Literal[2], Literal[2]]
print(cube)

SIDE = Literal[4]
arr : NDArray[Tuple[Literal[2], Literal[2]], np.int256] = np.zeros(shape=(2,2,2), dtype=np.int256)
# Uh oh the shapes won't match!

square: NDArray[Shape2D[SIDE, SIDE], np.float64] = rand_normal_matrix((3,3))

print(square)