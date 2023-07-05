# link: https://taoa.io/posts/Shape-typing-numpy-with-pyright-and-variadic-generics

import numpy as np

from numpy.typing import NDArray

from typing import Tuple, TypeVar, Literal

# Generic dimension sizes types

T1 = TypeVar("T1", bound=int)

T2 = TypeVar("T2", bound=int)

T3 = TypeVar("T3", bound=int)

# Dimension types represented as typles

Shape = Tuple

Shape1D = Shape[T1]

Shape2D = Shape[T1, T2]

Shape3D = Shape[T1, T2, T3]

ShapeND = Shape[T1, ...]

ShapeNDType = TypeVar("ShapeNDType", bound=ShapeND)


def rand_normal_matrix(shape: ShapeNDType) -> NDArray[ShapeNDType, np.float64]:

    """Return a random ND normal matrix."""

    return np.random.standard_normal(size=shape)


# Yay correctly typed 2x2x2 cube!

LENGTH = Literal[2]

cube: NDArray[Shape3D[LENGTH, LENGTH, LENGTH], np.float64] = rand_normal_matrix((2,2,2))

print(cube)

SIDE = Literal[4]


# Uh oh the shapes won't match!

square: NDArray[Shape2D[SIDE, SIDE], np.float64] = rand_normal_matrix((3,3))

print(square)