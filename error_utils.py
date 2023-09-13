from log_utils import logger_print

# from beartype import beartype
from typing import Union
import traceback
import sys

# @beartype
class ErrorManager:
    """
    Manage exceptions and errors.

    Can be used in `with` statements to automate such management, which behavior can be configured by setting `suppress_error` and `suppress_exception` arguments.

    Args:
    
    suppress_error:bool: If suppressed, don't raise exception if having error messages
    suppress_exception:bool: If suppressed, don't suppress exception raised by program
    default_error:str: The default error message to display if any error occurs during execution

    """
    def __init__(
        self,
        suppress_error: bool = False,
        suppress_exception: bool = False,
        default_error: Union[str, None] = None,
    ):

        self.errors = []
        self.suppress_error = suppress_error
        self.suppress_exception = suppress_exception
        self.default_error = default_error

    def __bool__(self):
        return len(self.errors) > 0

    @property
    def has_error(self):
        return bool(self)

    @property
    def has_exception(self):
        last_exc = sys.exc_info()
        return last_exc[0] is not None

    def append(self, error: str):
        self.errors.append(error)

    def clear(self):
        self.errors = []
        self.default_error = None

    def format_error(self, clear=True, join: str = "\n"):
        error_msg = join.join(
            self.errors
            + ([self.default_error] if (self and self.default_error) else [])
        )
        if clear:
            self.clear()
        return error_msg

    def raise_if_any(self):
        if self.errors:
            self.print_if_any()
            raise Exception(self.format_error())

    def print_if_any(self):
        if self.errors:
            logger_print(self.format_error())
            return True
        return False

    def __enter__(self):
        self.raise_if_any()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and not self.suppress_error:
            self.raise_if_any()
        else:
            self.print_if_any()
        
        if self.has_exception:
            traceback_exc = traceback.format_exc()
            logger_print(traceback_exc)
        return True if self.suppress_exception else None

    def __str__(self):
        return self.format_error(clear=False)

    def __repr__(self):
        return self.format_error(clear=False)

    def __len__(self):
        return len(self.errors)

    def __iter__(self):
        return iter(self.errors)


if __name__ == "__main__":
    # test this!
    with ErrorManager() as em:
        # raise Exception("before append")
        em.append('abc')
        raise Exception("after append")
