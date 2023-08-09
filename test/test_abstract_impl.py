from abc import ABC, abstractmethod

class abs_class(ABC):
    def method(self):
        return self._method_impl()
    @abstractmethod
    def _method_impl(self):
        ...

class impl_class(abs_class):
    def _method_impl(self):
        