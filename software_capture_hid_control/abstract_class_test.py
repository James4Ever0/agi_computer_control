from abc import ABC, abstractmethod, ABCMeta

# class Example(ABC):
class Example(ABC):
    @abstractmethod
    def a(self):
        print('a')
        
    @abstractmethod
    def b(self):
        print('a')

e = Example()