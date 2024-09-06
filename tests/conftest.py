from pathlib import Path
from typing import Optional

import pytest

from qxmt import (
    DatasetConfig,
    DeviceConfig,
    EvaluationConfig,
    ExperimentConfig,
    FeatureMapConfig,
    KernelConfig,
    ModelConfig,
    PathConfig,
)

DEFAULT_DATASET_CONFIG = DatasetConfig(
    type="generate",
    path=PathConfig(data=Path("data"), label=Path("label")),
    random_seed=42,
    test_size=0.2,
    features=None,
    raw_preprocess_logic=None,
    transform_logic=None,
)
DEFAULT_DEVICE_CONFIG = DeviceConfig(platform="pennylane", name="default.qubit", n_qubits=2, shots=None)
DEFAULT_FEATUREMAP_CONFIG = FeatureMapConfig(
    module_name="qxmt.feature_maps.pennylane", implement_name="ZZFeatureMap", params={"reps": 2}
)
DEFAULT_KERNEL_CONFIG = KernelConfig(module_name="qxmt.kernels.pennylane", implement_name="FidelityKernel", params={})
DEFAULT_MODEL_CONFIG = ModelConfig(name="qsvm", file_name="model.pkl", params={"C": 1.0, "gamma": 0.05})
DEFAULT_EVALUATION_CONFIG = EvaluationConfig(default_metrics=["accuracy", "precision", "recall", "f1_score"])


@pytest.fixture(scope="function")
def experiment_config(**kwargs: dict) -> ExperimentConfig:
    default_values = {
        "path": ".",
        "description": "test",
        "dataset": DEFAULT_DATASET_CONFIG,
        "device": DEFAULT_DEVICE_CONFIG,
        "feature_map": DEFAULT_FEATUREMAP_CONFIG,
        "kernel": DEFAULT_KERNEL_CONFIG,
        "model": DEFAULT_MODEL_CONFIG,
        "evaluation": DEFAULT_EVALUATION_CONFIG,
    }
    default_values.update(kwargs)
    return ExperimentConfig(**default_values)
