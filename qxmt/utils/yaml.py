import importlib
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import yaml


def load_yaml_config(file_path: str | Path) -> dict:
    """Load yaml from configuration file.

    Args:
        file_path (str | Path): path to the configuration file

    Returns:
        dict: configuration
    """
    if Path(file_path).suffix != ".yaml":
        raise ValueError("The configuration file must be a yaml file.")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"File '{file_path}' not found.")

    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def load_function_from_yaml(config: dict) -> Callable:
    """Extract function from yaml configuration

    Args:
        config (dict): configuration of the function

    Raises:
        ModuleNotFoundError: module not found in the path
        AttributeError: function not found in the module

    Returns:
        Callable[..., Any]: A function with parameters ready to be called.
    """
    module_name = config.get("module_name", None)
    function_name = config.get("function_name", None)
    params = config.get("params", None)

    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
    except ModuleNotFoundError:
        raise ImportError(f"Module '{module_name}' not found.")
    except AttributeError:
        raise AttributeError(f"Function '{function_name}' not found in module '{module_name}'.")

    @wraps(func)
    def callable_func(*args: Any, **kwargs: Any) -> Callable:
        """Wrapper function that applies the params to the extracted function."""
        if params is not None:
            combined_params = {**params, **kwargs}  # Merge YAML params with any additional kwargs
            return func(*args, **combined_params)
        else:
            return func(*args, **kwargs)

    # Set type hints for the wrapper function
    if hasattr(func, "__annotations__"):
        callable_func.__annotations__ = func.__annotations__

    return callable_func
