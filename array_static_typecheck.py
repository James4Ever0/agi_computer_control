# link: https://taoa.io/posts/Shape-typing-numpy-with-pyright-and-variadic-generics
# PEP 646: https://peps.python.org/pep-0646/

# mypy --enable-incomplete-feature=Unpack --enable-incomplete-feature=TypeVarTuple  array_static_typecheck.py

from typing import TypeVar, Generic, NewType, Literal
from typing_extensions import TypeVarTuple, Unpack, Self

DType = TypeVar('DType')
Shape = TypeVarTuple('Shape')

class Array(Generic[DType, Unpack[Shape]]):

    def __abs__(self) -> Self: ...

    def __add__(self, other: Self) -> Self: ...


Height = NewType('Height', int)
Width = NewType('Width', int)
x: Array[Height, Width] = Array()
y: Array[Literal[1], Literal[1]] = Array()
z = abs(y)