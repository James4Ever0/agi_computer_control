# link: https://taoa.io/posts/Shape-typing-numpy-with-pyright-and-variadic-generics
# PEP 646: https://peps.python.org/pep-0646/

# mypy --enable-incomplete-feature=Unpack --enable-incomplete-feature=TypeVarTuple  array_static_typecheck.py

from typing import TypeVar, Generic, NewType, Literal
from typing_extensions import TypeVarTuple, Unpack, Self, Annotated

DType = TypeVar("DType")
Shape = TypeVarTuple("Shape")


class Array(Generic[DType, Unpack[Shape]]):
    def __abs__(self) -> Self:
        ...

    def special_ops(self, a: Annotated[int, 2, 3]) -> Annotated[int, 1, 2]:
        ...

    def __add__(self, other: Self) -> Self:
        ...


Height = NewType("Height", int)
Width = NewType("Width", int)
x: Array[float, Height, Width] = Array()
y: Array[float, Literal[1], Literal[1]] = Array()
z = abs(y)

h = x + y

a0: Annotated[float, 1, 2] = 1
x.special_ops(a0)  # annotated will not be checked here.


annotated_array: "Annotated[1,1,1]"