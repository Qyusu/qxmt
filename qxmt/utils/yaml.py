import importlib
import types
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import yaml

from qxmt import ExperimentConfig


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


def load_object_from_yaml(config: dict, dynamic_params: dict = {}) -> Any:
    """Extract object from yaml configuration and convert it to a function or class instance.

    Args:
        config (dict): configuration of the object
        dynamic_params (dict, optional): dynamic parameters to be passed to the object. Defaults to {}.

    Raises:
        ModuleNotFoundError: module not found in the path
        AttributeError: object not found in the module
        TypeError: object is not a class or function

    Returns:
        Any: function or instance of the extracted object
    """
    module_name = config.get("module_name", None)
    object_name = config.get("implement_name", None)
    params = config.get("params", {})
    if params is None:
        params = dynamic_params
    else:
        params = params | dynamic_params

    try:
        module = importlib.import_module(module_name)
        obj = getattr(module, object_name)
    except ModuleNotFoundError:
        raise ImportError(f"Module '{module_name}' not found.")
    except AttributeError:
        raise AttributeError(f"Object '{object_name}' not found in module '{module_name}'.")

    if isinstance(obj, types.FunctionType):
        # Extracted object is a function
        @wraps(obj)
        def callable_func(*args: Any, **kwargs: Any) -> Callable:
            """Wrapper function that applies the params to the extracted function."""
            combined_params = {**params, **kwargs}  # Merge YAML params with any additional kwargs
            return obj(*args, **combined_params)

        # Set type hints for the wrapper function
        if hasattr(obj, "__annotations__"):
            callable_func.__annotations__ = obj.__annotations__

        return callable_func
    elif isinstance(obj, type):
        # Extracted object is a class instance
        return obj(**params)
    else:
        raise TypeError(f"'{object_name}' is not a class or function.")


def save_experiment_config_to_yaml(
    config: ExperimentConfig,
    save_path: str | Path,
    delete_path: bool = False,
) -> None:
    """Save the experiment configuration to a yaml file.

    Args:
        config (ExperimentConfig): experiment configuration
        save_path (str | Path): path to save the yaml file
    """
    config_dict = config.model_dump()
    if delete_path:
        del config_dict["path"]
    config_dict["dataset"]["path"]["data"] = str(config_dict["dataset"]["path"]["data"])
    config_dict["dataset"]["path"]["label"] = str(config_dict["dataset"]["path"]["label"])

    with open(save_path, "w") as file:
        yaml.dump(config_dict, file)
