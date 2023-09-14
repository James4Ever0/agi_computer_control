class MyDecorator:
    def __init__(self, func):
        self.func = func
        self.invoke_decorated_function()

    def invoke_decorated_function(self):
        # Code to execute after the 'def' block of the decorated function
        print("Executing code right after the 'def' block")
        self.func()


# Decorate a function with the decorator
@MyDecorator
def my_function():
    print("Executing the decorated function")


print("-" * 40)


class MyMeta(type):
    def __new__(cls, name, bases, attrs):
        # Code to execute right after the 'def' block of the decorated function
        print("Executing code right after the 'def' block")

        return super().__new__(cls, name, bases, attrs)


# Define the metaclass for the decorator
class MyDecorator(metaclass=MyMeta):
    def __call__(self, func):
        print("Executing code before the function is invoked")

        def wrapper(*args, **kwargs):
            # Code to execute before invoking the decorated function

            # Invoke the decorated function
            return func(*args, **kwargs)

        return wrapper


# Decorate a function with the decorator
@MyDecorator()
def my_function():
    print("Executing the decorated function")


print("-" * 20)


def my_decorator(func):
    print("Executing code before the function is invoked.")
    print("Function name:", func.__name__)
    def wrapper(*args, **kwargs):
        # Code to execute before the invocation of the decorated function

        # Invoke the decorated function
        return func(*args, **kwargs)

    # Return the wrapper function
    return wrapper


# Decorate a function with the decorator
@my_decorator
def my_function():
    print("Executing the decorated function.")
