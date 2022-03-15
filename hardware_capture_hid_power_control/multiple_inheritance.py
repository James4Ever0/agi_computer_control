from types import MethodType, MethodWrapperType


class A:
    def a(self):
        print("a")

    def b(self):
        print("a.b method")
        self.a()
        super().b()
        print("a.super()", super())
        print("super.b", super().b)
        my_b = super().b
        a_sup = super()
        #     breakpoint()
        print("a.super init?", init := getattr(super(), "__init__", None))
        print(dir(init), type(init), dir(type(init)))
        print("CALLING A.SUPER.INIT")
        super().__init__()
        print(
            type(super().__init__),
            isinstance(super().__init__, MethodType),
            isinstance(super().__init__, MethodWrapperType),
        )

    def d(self):
        print("a.d")


class C:
    def __init__(self):
        print("C.INIT")
        #   super().__init__(a=1)
        print(
            type(super().__init__),
            isinstance(super().__init__, MethodWrapperType),
            isinstance(super().__init__, MethodType),
        )

    def d(self):
        print("c.d")

    def b(self):
        print("c.b method")
        # super().b()
        print("c.super()", super())
        print(getattr(super(), "b", None))
        c_sup = super()
        # breakpoint()
        print("c.super init?", init := getattr(super(), "__init__", None))
        print(dir(init), type(init), dir(type(init)))

    def c(self):
        print("c")


class B(A, C):
    def __init__(self):
        print("CALLING B.INIT")
        self.A = super().__init__()

    def b(self):
        print("b.b method")
        print("b.super()", super())
        super().b()
        #     super().b()
        self.c()
        super(A,self).d()
        super(C,self).d()

    def a(self):
        print("override")


# B().b()
# c.b not shown up!

# b.b method
# a.b method
# override
# c

class Parent1:
    def method(self):
        print("Parent1 method")

class Parent2:
    def method(self):
        print("Parent2 method")

class Child(Parent1, Parent2):
    def method(self):
        super(Parent1, self).method()
        super(Parent2, self).method()

c = Child()
c.method()
