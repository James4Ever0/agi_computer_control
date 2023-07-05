# link: https://taoa.io/posts/Shape-typing-numpy-with-pyright-and-variadic-generics
# PEP 646: https://peps.python.org/pep-0646/



from typing import TypeVar, Generic
from typing_extensions import TypeVarTuple

DType = TypeVar('DType')
Shape = TypeVarTuple('Shape')

class Array(Generic[DType, *Shape]):

    def __abs__(self) -> Array[DType, *Shape]: ...

    def __add__(self, other: Array[DType, *Shape]) -> Array[DType, *Shape]: ...

