import time
from typing import Any, Callable


def notify_long_running(func: Callable) -> Callable:
    """Decorator to notify the user that a function is running."""

    def wrapper(*args: dict, **kwargs: Any) -> Any:
        print(f'Executing "{func.__name__}". This may take some time...')
        result = func(*args, **kwargs)
        return result

    return wrapper


def retry_on_exception(retries: int, delay: float) -> Callable:
    """Decorator to retry a function if an exception is raised."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: dict, **kwargs: Any) -> None:
            attempt = 0
            while attempt < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= retries:
                        raise e
                    time.sleep(delay)

        return wrapper

    return decorator
