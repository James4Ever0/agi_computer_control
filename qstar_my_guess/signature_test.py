from typing_extensions import overload
from overtake import overtake  # type: ignore


@overload
def func(a: int) -> None:
    print("int", a)


@overload
def func(a: str) -> None:
    print("str", a)


@overload
def func(a: str, b: int) -> None:
    print("str & int", a, b)


@overtake(runtime_type_checker="beartype")
def func(a, b=1):
    ...


def c():
    print("a")


# Example usage
c()
func(10)
func("Hello")
func("Hello", 1)
func("Hello", "World")  # failed to type check.
func("Hello", 1)
c()

# func([]) # beartype failed to check this
