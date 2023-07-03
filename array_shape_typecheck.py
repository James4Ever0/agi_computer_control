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
# from jax import Array  # type: ignore

# import jaxtyping  # type: ignore
# from typing_extensions import TypeAlias
from typing import cast

# mTypeAlias: TypeAlias = jaxtyping.Float[Array, "dim1 dim2"]
# arr3 = cast(mTypeAlias, np.array([[1, 2, 3]]))
# # arr3: mTypeAlias = np.array([[1, 2, 3]])
# arr4: jaxtyping.Float[Array, "dim1 dim3"] = np.array([[1, 2, 3, 5]])


# @beartype.beartype
# def add2(a: mTypeAlias, b: mTypeAlias) -> mTypeAlias:
#     return a + b


# # arr5 = add2(arr3, arr4)
# arr5 = add2(arr3, arr3)  # still not working.


# from typing import TypeVar, Generic
# from typing_extensions import TypeVarTuple, Unpack

# DType = TypeVar("DType")
# Shape = TypeVarTuple("Shape")


# class Array(Generic[DType, Unpack[Shape]]):
#     def __abs__(self) -> Array[DType, Unpack[Shape]]:
#         ...

#     def __add__(
#         self, other: Array[DType, Unpack[Shape]]
#     ) -> Array[DType, Unpack[Shape]]:
#         ...


# from typing import Literal

# arr9 = cast(Array[int, Literal[1], Literal[3]], np.array([[1, 2, 3]]))
# arr10 = cast(Array[int, Literal[1], Literal[4]], np.array([[1, 2, 3, 4]]))

# arr11 = arr9 + arr10  # checked!
# arr11 = arr9+arr9

from typing_extensions import Annotated

myType = Annotated[np.ndarray, 20, 30]
myArr: myType = np.zeros((20, 30))
# how to get that annotated value?
# print('ANNOTATION?',myArr.__annotations__)
print(__annotations__)
# {'arr': NDArray[Shape['5, 10'], Any], 'arr2': NDArray[Shape['10, 5'], Any], 'myArr': typing_extensions.Annotated[numpy.ndarray, 20, 30]}