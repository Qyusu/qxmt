from pathlib import Path

import pytest

from qxmt.utils import load_class_from_yaml, load_function_from_yaml, load_yaml_config


class TestLoadYamlConfig:
    def test_load_yaml_config(self, tmp_path: Path) -> None:
        config = {
            "description": "This is a test configuration",
            "dataset": "dummy",
            "device": {"platform": "pennylane"},
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as file:
            file.write(f"{config}")
        loaded_config = load_yaml_config(config_file)
        assert loaded_config == config

    def test_load_yaml_config_invalid_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.txt"
        with open(config_file, "w") as file:
            file.write("This is not a yaml file.")
        with pytest.raises(ValueError):
            load_yaml_config(config_file)

    def test_load_yaml_config_file_not_found(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        with pytest.raises(FileNotFoundError):
            load_yaml_config(config_file)


def simple_function(x: int, y: int) -> int:
    return x + y


class TestLoadFunctionFromYaml:
    def test_no_params(self) -> None:
        config = {
            "module_name": __name__,
            "function_name": "simple_function",
            "params": None,
        }
        func = load_function_from_yaml(config)
        assert func(x=1, y=1) == 2

    def test_with_params(self) -> None:
        config = {
            "module_name": __name__,
            "function_name": "simple_function",
            "params": {"x": 1},
        }
        func = load_function_from_yaml(config)
        assert func(y=1) == 2

    def test_module_not_found(self) -> None:
        config = {
            "module_name": "not_exist",
            "function_name": "simple_function",
            "params": None,
        }
        with pytest.raises(ImportError):
            load_function_from_yaml(config)

    def test_function_not_found(self) -> None:
        config = {
            "module_name": __name__,
            "function_name": "not_exist",
            "params": None,
        }
        with pytest.raises(AttributeError):
            load_function_from_yaml(config)


class SimpleClass:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def add(self) -> int:
        return self.x + self.y


class TestLoadClassFromYaml:
    def test_with_yaml(self) -> None:
        config = {
            "module_name": __name__,
            "implement_name": "SimpleClass",
            "params": {"x": 1, "y": 1},
        }
        instance = load_class_from_yaml(config)
        assert instance.x == 1
        assert instance.y == 1
        assert instance.add() == 2

    def test_with_dynamic_params(self) -> None:
        config = {
            "module_name": __name__,
            "implement_name": "SimpleClass",
            "params": {"x": 1},
        }
        dynamic_params = {"y": 1}
        instance = load_class_from_yaml(config, dynamic_params)
        assert instance.x == 1
        assert instance.y == 1
        assert instance.add() == 2

    def test_module_not_found(self) -> None:
        config = {
            "module_name": "not_exist",
            "implement_name": "simple_function",
            "params": {"x": 1, "y": 1},
        }
        with pytest.raises(ImportError):
            load_class_from_yaml(config)

    def test_class_not_found(self) -> None:
        config = {
            "module_name": __name__,
            "implement_name": "not_exist",
            "params": {"x": 1, "y": 1},
        }
        with pytest.raises(AttributeError):
            load_class_from_yaml(config)

    def test_not_class(self) -> None:
        config = {
            "module_name": __name__,
            "implement_name": "simple_function",
            "params": {"x": 1, "y": 1},
        }
        with pytest.raises(TypeError):
            load_class_from_yaml(config)

    def test_over_params(self) -> None:
        config = {
            "module_name": __name__,
            "implement_name": "SimpleClass",
            "params": {"x": 1, "y": 1, "z": 1},
        }
        with pytest.raises(TypeError):
            load_class_from_yaml(config)

    def test_lack_params(self) -> None:
        config = {
            "module_name": __name__,
            "implement_name": "SimpleClass",
            "params": {"x": 1},
        }
        with pytest.raises(TypeError):
            load_class_from_yaml(config)
