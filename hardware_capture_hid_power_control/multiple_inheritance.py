class A:
    def a(self):
            print("a")
    def b(self):
            print("a.b method")
            self.a()

a
class C:
    def  b(self):
        print("c.b method")

class B(A):
    def __init__(self):
            self.A = super().__init__()
    def b(self):
            print('b.b method')
            super().b()
    def a(self):
            print('override')

B().b()