from typing import Any, Callable


def notify_long_running(func: Callable) -> Callable:
    def wrapper(*args: dict, **kwargs: dict) -> Any:
        print(f'Executing "{func.__name__}". This may take some time...')
        result = func(*args, **kwargs)
        return result

    return wrapper
