from importlib.metadata import PackageNotFoundError, version

from qxmt.configs import (
    AnsatzConfig,
    DatasetConfig,
    DeviceConfig,
    EvaluationConfig,
    ExperimentConfig,
    FeatureMapConfig,
    FileConfig,
    GenerateDataConfig,
    GlobalSettingsConfig,
    HamiltonianConfig,
    KernelConfig,
    ModelConfig,
    SplitConfig,
    TFDSConfig,
)
from qxmt.exceptions import (
    AmazonBraketSettingError,
    DeviceSettingError,
    ExperimentNotInitializedError,
    ExperimentRunSettingError,
    ExperimentSettingError,
    IBMQSettingError,
    InputShapeError,
    InvalidFileExtensionError,
    InvalidModelNameError,
    InvalidPlatformError,
    InvalidQunatumDeviceError,
    JsonEncodingError,
    ModelSettingError,
    ReproductionError,
)
from qxmt.experiment.experiment import Experiment

__all__ = [
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


try:
    __version__ = version("qxmt")
except PackageNotFoundError:
    __version__ = "0.0.0"
