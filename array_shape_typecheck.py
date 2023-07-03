from typing_extensions import reveal_type
import numpy as np

# from typing import Tuple
# from numpy.typing import NDArray, DTypeLike
from nptyping import NDArray, Shape, Float  # type: ignore
from typing import Any

M = 5
N = 10
arr: NDArray[Shape["5, 10"], Any] = np.zeros((M, N))
arr2: NDArray[Shape["10, 5"], Any] = np.zeros((N, M))
import beartype  # type:ignore


@beartype.beartype # check before run?
def add_arrays(
    arr1: NDArray[Shape["5, 10"], Any], arr2: NDArray[Shape["5, 10"], Any]
) -> NDArray[Shape["5, 10"], Any]:
    result = arr1 + arr2
    return result


# myarr = add_arrays(arr, arr)  # no issue?
myarr = add_arrays(arr, arr2) # only beartype shows issue.
reveal_type(myarr)


import jaxtyping

arr3:jaxtyping.Float[jaxtyping.Array