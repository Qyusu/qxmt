from typing import Any

import pytest

from qxmt import (
    DatasetConfig,
    DeviceConfig,
    EvaluationConfig,
    ExperimentConfig,
    FeatureMapConfig,
    GenerateDataConfig,
    GlobalSettingsConfig,
    KernelConfig,
    ModelConfig,
    SplitConfig,
)

DEFAULT_GLOBAL_SETTINGS = GlobalSettingsConfig(
    random_seed=42,
    task_type="classification",
)

DEFAULT_DATASET_CONFIG = DatasetConfig(
    generate=GenerateDataConfig(generate_method="linear"),
    openml=None,
    file=None,
    random_seed=DEFAULT_GLOBAL_SETTINGS.random_seed,
    split=SplitConfig(train_ratio=0.8, validation_ratio=0.0, test_ratio=0.2, shuffle=True),
    features=None,
    raw_preprocess_logic=None,
    transform_logic=None,
)
DEFAULT_DEVICE_CONFIG = DeviceConfig(platform="pennylane", name="default.qubit", n_qubits=2, shots=None)
SHOTS_DEVICE_CONFIG = DeviceConfig(platform="pennylane", name="default.qubit", n_qubits=2, shots=5)
DEFAULT_FEATUREMAP_CONFIG = FeatureMapConfig(
    module_name="qxmt.feature_maps.pennylane", implement_name="ZZFeatureMap", params={"reps": 2}
)
DEFAULT_KERNEL_CONFIG = KernelConfig(module_name="qxmt.kernels.pennylane", implement_name="FidelityKernel", params={})
DEFAULT_MODEL_CONFIG = ModelConfig(name="qsvc", params={"C": 1.0, "gamma": 0.05})
DEFAULT_EVALUATION_CONFIG = EvaluationConfig(default_metrics=["accuracy", "precision", "recall", "f1_score"])


@pytest.fixture(scope="function")
def experiment_config(**kwargs: Any) -> ExperimentConfig:
    default_values = {
        "path": ".",
        "description": "test",
        "global_settings": DEFAULT_GLOBAL_SETTINGS,
        "dataset": DEFAULT_DATASET_CONFIG,
        "device": DEFAULT_DEVICE_CONFIG,
        "feature_map": DEFAULT_FEATUREMAP_CONFIG,
        "kernel": DEFAULT_KERNEL_CONFIG,
        "model": DEFAULT_MODEL_CONFIG,
        "evaluation": DEFAULT_EVALUATION_CONFIG,
    }
    default_values.update(kwargs)
    return ExperimentConfig(**default_values)


@pytest.fixture(scope="function")
def shots_experiment_config(**kwargs: Any) -> ExperimentConfig:
    default_values = {
        "path": ".",
        "description": "test",
        "global_settings": DEFAULT_GLOBAL_SETTINGS,
        "dataset": DEFAULT_DATASET_CONFIG,
        "device": SHOTS_DEVICE_CONFIG,
        "feature_map": DEFAULT_FEATUREMAP_CONFIG,
        "kernel": DEFAULT_KERNEL_CONFIG,
        "model": DEFAULT_MODEL_CONFIG,
        "evaluation": DEFAULT_EVALUATION_CONFIG,
    }
    default_values.update(kwargs)
    return ExperimentConfig(**default_values)
