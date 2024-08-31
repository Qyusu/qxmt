from pathlib import Path

import pytest

from qxmt.utils import extract_function_from_yaml, load_yaml_config


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


class TestExtractFunctionFromYaml:
    def test_extract_function_from_yaml(self) -> None:
        config = {
            "module_name": "math",
            "function_name": "sqrt",
            "params": [4],
        }
        func = extract_function_from_yaml(config)
        assert func(*config["params"]) == 2

    def test_extract_function_from_yaml_module_not_found(self) -> None:
        config = {
            "module_name": "mathh",
            "function_name": "sqrt",
            "params": [4],
        }
        with pytest.raises(ImportError):
            extract_function_from_yaml(config)

    def test_extract_function_from_yaml_function_not_found(self) -> None:
        config = {
            "module_name": "math",
            "function_name": "sqrtt",
            "params": [4],
        }
        with pytest.raises(AttributeError):
            extract_function_from_yaml(config)
