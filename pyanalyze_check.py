# from typing import TYPE_CHECKING
import typing
typing.TYPE_CHECKING = True
from pyanalyze.extensions import *

# from pyanalyze.extensions import CustomCheck, Value, CanAssignContext, CanAssign
import pyanalyze
from typing_extensions import Annotated


class LiteralOnly(CustomCheck):
    def can_assign(self, value: Value, ctx: CanAssignContext) -> CanAssign:
        for subval in pyanalyze.value.flatten_values(value):
            if not isinstance(subval, pyanalyze.value.KnownValue):
                return pyanalyze.value.CanAssignError("Value must be a literal")
        return {}


def func(arg: Annotated[str, LiteralOnly()]) -> None:
    ...


def some_call():
    # it is actually running this. damn it!
    print("CALLING FUNCTION")
    return "abc"


func("x")  # ok
func(str(some_call()))  # error
