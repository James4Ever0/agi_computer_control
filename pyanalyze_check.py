# from typing import TYPE_CHECKING
# import typing
# typing.TYPE_CHECKING = True
import dill
import pytest
import ast
import inspect
from pyanalyze.value import Value, CanAssign, CanAssignContext
from pyanalyze.extensions import CustomCheck

# from pyanalyze.extensions import CustomCheck, Value, CanAssignContext, CanAssign
import pyanalyze
from typing_extensions import Annotated
import rich


class LiteralOnly(CustomCheck):
    def can_assign(self, value: Value, ctx: CanAssignContext) -> CanAssign:
        rich.print(value.__dict__)
        rich.print(ctx.__dict__)
        breakpoint()
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


def anotherfunc():
    func("x")  # ok
    func(str(some_call()))  # error


# actually will not run the code, only if putting in between definitions.
# anotherfunc() # will run
if __name__ == "__main__":
    anotherfunc()  # will not run


@pytest.mark.parametrize("mylambda", [lambda x: x == 0, lambda x: x < 0])
def test_0(mylambda):
    # a = lambda x: x<0
    # print(dill.dumps(mylambda))
    # print(dill.source.dumpsource(mylambda))
    ...
    # a_source = inspect.getsource(mylambda)  # full source being dumped. not the lambda expression alone.
    # print(a_source)
    # tree = ast.parse(a_source)
