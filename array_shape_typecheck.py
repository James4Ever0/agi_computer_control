from typing_extensions import reveal_type
import numpy as np
# from typing import Tuple
# from numpy.typing import NDArray, DTypeLike

M = 5
N = 10
arr = np.zeros((M, N))
reveal_type(arr)
