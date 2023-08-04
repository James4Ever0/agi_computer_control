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
            print("a.super init?", init:=getattr(super(), '__init__', None))
            print(dir(init), type(init),dir(type(init)))
            print("CALLING A.SUPER.INIT")
            super().__init__()
            print(type(super().__init__), isinstance(super().__init__, MethodType), isinstance(super().__init__, MethodWrapperType))


class C:
    def __init__(self):
          print("C.INIT")
        #   super().__init__(a=1)
          print(type(super().__init__), isinstance(super().__init__, MethodWrapperType), isinstance(super().__init__, MethodType))
    def b(self):
        print("c.b method")
        # super().b() 
        print("c.super()", super())
        print(getattr(super(), 'b', None))
        c_sup = super()
        # breakpoint()
        print("c.super init?", init:=getattr(super(), '__init__', None))
        print(dir(init), type(init), dir(type(init)))
    def c(self):
        print('c')

class B(A,C):
    def __init__(self):
            print("CALLING B.INIT")
            self.A = super().__init__()
    def b(self):
            print('b.b method')
            print("b.super()", super())
            super().b()
        #     super().b()
            self.c()
    def a(self):
            print('override')

B().b()
# c.b not shown up!

# b.b method
# a.b method
# override
# c