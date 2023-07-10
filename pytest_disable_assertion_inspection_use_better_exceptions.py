# commandline:
# env BETTER_EXCEPTIONS=1 python3 -m pytest --full-capture --assert=plain pytest_disable_assertion_inspection_use_better_exceptions.py
# env BETTER_EXCEPTIONS=1 python3 -m pytest pytest_disable_assertion_inspection_use_better_exceptions.py

from pytest import MonkeyPatch
import numpy as np
import better_exceptions

# # import unittest
from pytest import ExceptionInfo

# def max_traceback_limit(tb, max_limit = 3):
#     if getattr(tb, 'tb_next',None):
#         if max_limit == 0:
#             tb.tb_next = None
#         else:
#             max_traceback_limit(tb.tb_next, max_limit = max_limit-1)

# import rich


def patch(exc_info, exprinfo):
    tb = exc_info[2]
    # max_traceback_limit(tb)
    # traceback is staring from the root cause. deal it in the end.
    # rich.print(tb)
    # breakpoint()
    cls = ExceptionInfo
    textlist = better_exceptions.format_exception(
        exc=exc_info[0], value=exc_info[1], tb=tb
    )
    # textlist = better_exceptions.format_exception(*exc_info)
    text = "".join(textlist)
    keyword = "in pytest_pyfunc_call"
    text = text.split("\n")
    last_index = -20
    for i, t in enumerate(text):
        if keyword in t:
            last_index = i
            break
    text = text[last_index:]
    text = "\n".join(text)
    print()
    print(text)  # great. this could be the hook.
    return cls(exc_info, text, _ispytest=True)


ExceptionInfo.from_exc_info = patch
# better_exceptions.hook()


def create_array():
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3, 4])
    c = a + b
    return c


import numpy


class marray:
    def __init__(self, content):
        print("CREATING ARRAY WITH CONTENT:", content)
        # how do you inspect that after patched the original method?
        # shall you return "None"
        # return "CREATED_ARRAY"


def test_mytest(monkeypatch: MonkeyPatch):
    # monkeypatch.setitem(numpy.__dict__, "array", marray) # patched!
    monkeypatch.setattr(numpy, "array", marray)
    a = 1
    b = {}
    create_array()
    # print(b[1])
    # assert b[2] == a


# if __name__ == "__main__":
#     test_mytest()
