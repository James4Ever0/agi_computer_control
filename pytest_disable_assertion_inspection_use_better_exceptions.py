# commandline:
# env BETTER_EXCEPTIONS=1 python3 -m pytest --full-capture --assert=plain pytest_disable_assertion_inspection_use_better_exceptions.py
# env BETTER_EXCEPTIONS=1 python3 -m pytest pytest_disable_assertion_inspection_use_better_exceptions.py

# import better_exceptions
# # import unittest

# better_exceptions.hook()

def test_mytest():
    a = 1
    b = {}
    # print(b[1])
    assert b[2] == a

if __name__ == "__main__":
    test_mytest()