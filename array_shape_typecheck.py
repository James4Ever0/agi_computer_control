from typing_extensions import reveal_type
import numpy as np

# from typing import Tuple
# from numpy.typing import NDArray, DTypeLike
from nptyping import NDArray, Shape, Float

M = 5
N = 10
arr: NDArray[Shape["5, 10"], Float] = np.zeros((M, N))
arr2: NDArray[Shape["10, 5"], Float] = np.zeros((N, M))
# myarr = arr + arr2  # no issue?
reveal_type(arr)
