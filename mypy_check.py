import mypy.api as api

result = api.run(...)  # commandline args.

import tensorflow as tf

from tensor_annotations import axes # by deepmind.
import tensor_annotations.tensorflow as ttf
from typing import NewType
from typing_extensions import Annotated

uint8 = ttf.uint8
Batch, Time = Annotated[axes.Batch, 3], Annotated[axes.Time, 5] # problem is, how to share this along with function calls?

MyAxis = NewType("MyAxis", axes.Axis)

# from pycontract import contract

# @contract
# def my_function(a, b):
#     """ Function description.
#         :type a: int,>0
#         :type b: list[N],N>0
#         :rtype: list[N]
#     """
#     ...


# def sample_batch() -> ttf.Tensor2[uint8, Time, Batch]:
#     return tf.zeros((3, 5))

from typing_extensions import Annotated
from annotated_types import Gt, Len, Predicate

class MyClass:
    age: Annotated[int, Gt(18)]                         # Valid: 19, 20, ...
                                                        # Invalid: 17, 18, "19", 19.0, ...
    factors: list[Annotated[int, Predicate(is_prime)]]  # Valid: 2, 3, 5, 7, 11, ...
                                                        # Invalid: 4, 8, -2, 5.0, "prime", ...

    my_list: Annotated[list[int], Len(0, 10)]           # Valid: [], [10, 20, 30, 40, 50]
                                                        # Invalid: (1, 2), ["abc"], [0] * 20