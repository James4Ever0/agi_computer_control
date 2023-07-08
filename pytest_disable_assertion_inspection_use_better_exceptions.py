# commandline:
# env BETTER_EXCEPTIONS=1 python3 -m pytest --full-capture --assert=plain pytest_disable_assertion_inspection_use_better_exceptions.py
# env BETTER_EXCEPTIONS=1 python3 -m pytest pytest_disable_assertion_inspection_use_better_exceptions.py

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
        exc=exc_info[0], value=exc_info[1], tb=tb)
    # textlist = better_exceptions.format_exception(*exc_info)
    text = "".join(textlist)
    text = "\n".join(text.split("\n")[-10:])
    print(text)  # great. this could be the hook.
    return cls(exc_info, text, _ispytest=True)


ExceptionInfo.from_exc_info = patch
# better_exceptions.hook()


def test_mytest():
    a = 1
    b = {}
    print(b[1])
    # assert b[2] == a


if __name__ == "__main__":
    test_mytest()
