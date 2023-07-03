from typing_extensions import reveal_type
import numpy as np

# from typing import Tuple
# from numpy.typing import NDArray, DTypeLike
from nptyping import NDArray, Shape, Float # type: ignore
from typing import Any

M = 5
N = 10
arr: NDArray[Shape["5, 10"], Any] = np.zeros((M, N))
arr2: NDArray[Shape["10, 5"], Any] = np.zeros((N, M))
import beartype # type:ignore
@beartype.beartype
def add_arr(arr1: NDArray[Shape['5,10'], Any], arr2: NDArray[Shape['5,10'], Any]) -> NDArray[Shape['5,10'], Any]:
    myarr = arr1 + arr2  # no issue?
    return myarr
myarr = add_arr(arr, arr2) # no issue?
reveal_type(myarr)
