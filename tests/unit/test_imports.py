from importlib.metadata import PackageNotFoundError, version

from qxmt import __all__, __version__

EXPECTED_ALL = [
    "AmazonBraketSettingError",
    "AnsatzConfig",
    "DatasetConfig",
    "DeviceConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "FeatureMapConfig",
    "FileConfig",
    "GenerateDataConfig",
    "GlobalSettingsConfig",
    "HamiltonianConfig",
    "KernelConfig",
    "ModelConfig",
    "SplitConfig",
    "TFDSConfig",
    "DeviceSettingError",
    "ExperimentNotInitializedError",
    "ExperimentRunSettingError",
    "ExperimentSettingError",
    "IBMQSettingError",
    "InputShapeError",
    "InvalidFileExtensionError",
    "InvalidModelNameError",
    "InvalidPlatformError",
    "InvalidQunatumDeviceError",
    "JsonEncodingError",
    "ModelSettingError",
    "ReproductionError",
    "Experiment",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)


def test_version() -> None:
    try:
        expected_version = version("qxmt")
    except PackageNotFoundError:
        expected_version = "0.0.0"

    assert __version__ == expected_version
