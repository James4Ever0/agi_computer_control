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
# no annotation!
import beartype  # type:ignore


@beartype.beartype  # check before run?
def add_arrays(
    arr1: NDArray[Shape["5, 10"], Any], arr2: NDArray[Shape["5, 10"], Any]
) -> NDArray[Shape["5, 10"], Any]:
    result = arr1 + arr2
    return result


myarr = add_arrays(arr, arr)  # no issue?
# myarr = add_arrays(arr, arr2)  # only beartype shows issue.
reveal_type(myarr)

# from jaxtyping import Array
# not typechecking. import from jax.
from jax import Array # type: ignore

import jaxtyping  # type: ignore
from typing_extensions import TypeAlias
mTypeAlias: TypeAlias = jaxtyping.Float[Array, "dim1 dim2"]
arr3: mTypeAlias = np.array([[1, 2, 3]])
arr4: jaxtyping.Float[Array, "dim1 dim3"] = np.array([[1, 2, 3, 5]])
@beartype.beartype
def add2(a: mTypeAlias, b: mTypeAlias)-> mTypeAlias:
    return a + b


arr5 = add2(arr3, arr4)
