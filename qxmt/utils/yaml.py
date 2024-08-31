import importlib
from pathlib import Path
from typing import Callable

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


def extract_function_from_yaml(config: dict) -> Callable:
    """Extract function from yaml configuration

    Args:
        config (dict): configuration of the function

    Raises:
        ModuleNotFoundError: module not found in the path
        AttributeError: function not found in the module

    Returns:
        Callable: extracted function
    """
    module_name = config["module_name"]
    function_name = config["function_name"]
    params = config["params"]

    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
    except ModuleNotFoundError:
        raise ImportError(f"Module '{module_name}' not found.")
    except AttributeError:
        raise AttributeError(f"Function '{function_name}' not found in module '{module_name}'.")

    return func
