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
            print(dir(init), type(init))

class C:
    def __init__(self):
        ...
    def b(self):
        print("c.b method")
        # super().b() 
        print("c.super()", super())
        print(getattr(super(), 'b', None))
        c_sup = super()
        # breakpoint()
        print("c.super init?", init:=getattr(super(), '__init__', None))
        print(dir(init), type(init))
    def c(self):
        print('c')

class B(A,C):
    def __init__(self):
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