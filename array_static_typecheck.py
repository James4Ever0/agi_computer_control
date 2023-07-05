# link: https://taoa.io/posts/Shape-typing-numpy-with-pyright-and-variadic-generics
# PEP 646: https://peps.python.org/pep-0646/



from typing import TypeVar, Generic
from typing_extensions import TypeVarTuple, Unpack, Self

DType = TypeVar('DType')
Shape = TypeVarTuple('Shape')

class Array(Generic[DType, Unpack[Shape]]):

    def __abs__(self) -> Self[DType, Unpack[Shape]]: ...

    def __add__(self, other: Self[DType, Unpack[Shape]]) -> Self[DType, Unpack[Shape]]: ...

