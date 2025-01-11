from qxmt import __all__, __version__

EXPECTED_ALL = [
    "DatasetConfig",
    "DeviceConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "FeatureMapConfig",
    "FileConfig",
    "GenerateDataConfig",
    "GlobalSettingsConfig",
    "KernelConfig",
    "ModelConfig",
    "SplitConfig",
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
    assert __version__ == "0.4.1"
