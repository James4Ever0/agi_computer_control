from log_utils import logger_print

# from beartype import beartype
from typing import Union


# @beartype
# TODO: support custom exception/error handlers/formatters
class ExceptionManager:
    def __init__(
        self,
        suppress_error: bool = False,
        suppress_exception: bool = False,
        default_error: Union[str, None] = None,
    ):
        """
        Manage exceptions and errors.
        Can be used in `with` statements to automate management, which behavior can be configured by setting `suppress_error` and `suppress_exception` arguments.

        Args:
            suppress_error:bool: If suppressed, don't treat manual appended error messages as exception
            suppress_exception:bool: If suppressed, don't suppress exception raised by program

        """
        self.errors = []
        self.suppress_error = suppress_error
        self.suppress_exception = suppress_exception
        self.default_error = default_error

    def __bool__(self):
        return len(self.errors) > 0

    def has_exception(self):
        return bool(self)

    def append(self, error: str):
        if not isinstance(error, str):
            raise Exception("Expected error to be a string.\nPassed: " + error)
        self.errors.append(error)

    def clear(self):
        self.errors = []
        self.default_error = None

    def format_error(self, clear=True, join: str = "\n"):
        msgs = self.errors + (
            [self.default_error] if (self and (self.default_error is not None)) else []
        )
        error_msg = join.join(msgs)
        if clear:
            self.clear()
        return error_msg

    def raise_if_any(self):
        if self.errors:
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
        return True if self.suppress_exception else None

    def __str__(self):
        return self.format_error(clear=False)

    def __repr__(self):
        return self.format_error(clear=False)

    def __len__(self):
        return len(self.errors)

    def __iter__(self):
        return iter(self.errors)


exceptionManager = ExceptionManager(suppress_error=True)
