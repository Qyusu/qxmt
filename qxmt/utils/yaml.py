import copy
import importlib
import sys
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


def load_object_from_yaml(config: dict, dynamic_params: dict = {}, use_cache: bool = False) -> Any:
    """Extract object from yaml configuration and convert it to a function or class instance.

    Args:
        config (dict): configuration of the object
        dynamic_params (dict, optional): dynamic parameters to be passed to the object. Defaults to {}.
        use_cache (bool, optional): use cache when loading the module. Defaults to False.

    Raises:
        ModuleNotFoundError: module not found in the path
        AttributeError: object not found in the module
        TypeError: object is not a class or function

    Returns:
        Any: function or instance of the extracted object
    """
    module_name = config.get("module_name", None)
    object_name = config.get("implement_name", None)
    params = config.get("params", None)
    params = {} if params is None else params
    params.update(dynamic_params)

    try:
        if (not use_cache) and (module_name in sys.modules):
            # load module not use cache
            module = sys.modules[module_name]
            importlib.reload(module)
        else:
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
    delete_source_path: bool = True,
) -> None:
    """Save the experiment configuration to a yaml file.

    Args:
        config (ExperimentConfig): experiment configuration
        save_path (str | Path): path to save the configuration file
        delete_source_path (bool, optional): delete the source path in the configuration. Defaults to True.
    """
    save_config = copy.deepcopy(config)

    if save_config.dataset is not None and save_config.dataset.openml is not None:
        save_config.dataset.openml.save_path = str(save_config.dataset.openml.save_path)

    if save_config.dataset is not None and save_config.dataset.file is not None:
        save_config.dataset.file.data_path = str(save_config.dataset.file.data_path)
        save_config.dataset.file.label_path = str(save_config.dataset.file.label_path)

    config_dict = save_config.model_dump()
    if delete_source_path:
        del config_dict["path"]

    with open(save_path, "w") as file:
        yaml.dump(config_dict, file, sort_keys=False)
