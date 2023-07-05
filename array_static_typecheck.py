# link: https://taoa.io/posts/Shape-typing-numpy-with-pyright-and-variadic-generics

from typing import TypeVar, Any
from typing_extensions import TypeVarTuple

import numpy as np

ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)


# if TYPE_CHECKING or sys.version_info >= (3, 9):

ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)

Shape = TypeVarTuple("Shape")

_DType = np.dtype[ScalarType]

NDArray = np.ndarray[*Shape, np.dtype[ScalarType]]