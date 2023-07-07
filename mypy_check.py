import mypy.api as api

result = api.run(...) # commandline args.

import tensorflow as tf

from tensor_annotations import axes
import tensor_annotations.tensorflow as ttf
from typing import NewType
from typing_extensions import Annotated

uint8 = ttf.uint8
Batch, Time = Annotated[axes.Batch,3], Annotated[axes.Time, 5]

MyAxis = NewType("MyAxis", axes.Axis)

def sample_batch() -> ttf.Tensor2[uint8, Time, Batch]:
  return tf.zeros((3, 5))