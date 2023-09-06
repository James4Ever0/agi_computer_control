import func_timeout
import retrying
import functools
# let's not use multitasking

# timeout decorator for func with no arg/kwarg
timeout_func = lambda timeout: (
    lambda func: functools.partial(
        func_timeout.func_timeout, timeout=timeout, func=func
    )
)

retrying_timeout_func = lambda retry_max_count, timeout: (
    lambda func:
    # lambda func: multitasking.task(
    retrying.retry(stop_max_attempt_number=retry_max_count)(timeout_func(timeout)(func))
    # )
)

__all__ = ["timeout_func", "retrying_timeout_func"]