from abc import ABC, abstractmethod

class abs_class(ABC):
    def method(self):
        "doc here"
        return self._method_impl()
    @abstractmethod
    def _method_impl(self):
        ...

class impl_class(abs_class):
    def _method_impl(self):
        return 1
class impl_class2(abs_class):
    def non_relevant(self):
        ...
print(impl_class().method())
# print(impl_class2()) # error
# print(abs_class()) # error