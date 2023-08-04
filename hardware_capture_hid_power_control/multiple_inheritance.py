class A:
    def a(self):
            print("a")
    def b(self):
            print("a.b method")
            self.a()

class C:
    def b(self):
        print("c.b method")
    def c(self):
        print('c')

class B(A,C):
    def __init__(self):
            self.A = super().__init__()
    def b(self):
            print('b.b method')
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