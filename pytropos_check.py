# # pip install git+https://github.com/helq/pytropos
# # pytropos pytropos_check.py

# import numpy as np

# a = np.zeros((10, 6))

# m = 4 + 1
# n = 0 + 2

# if m == 5:
#     n = 1
# else:
#     m = 6

# b = np.ones((m, n))
# res = np.dot(a, b)  # fails here

# print(res)

# var = True

# if var:
#     b = np.zeros((3, 11))
#     res = np.dot(b, a)  # fails here

# print(res)

from hypothesis import given
from hypothesis.strategies import text
import mock
from pytest import MonkeyPatch, FixtureRequest
# from unittest.mock import MagicMock # replace class.

from unittest.mock import *
# mock.patch()

# common fixture!


def test_monkey(monkeypatch: MonkeyPatch, request: FixtureRequest):
    # now you finally have a better view on fixtures.
    monkeypatch.setitem


@given(s=text(), s0=text())
def test_myfun(s, s0):
    print(s, s0)  # lots of charactors.
