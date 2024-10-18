from pathlib import Path
from typing import Callable

import numpy as np
import pennylane as qml
import pytest

from qxmt import DatasetConfig, Experiment, GenerateDataConfig, SplitConfig
from qxmt.datasets import Dataset
from qxmt.devices import BaseDevice
from qxmt.kernels import BaseKernel
from qxmt.models import QSVC, BaseMLModel

DEVICE = BaseDevice(platform="pennylane", name="default.qubit", n_qubits=2, shots=None)


def empty_feature_map(x: np.ndarray) -> None:
    qml.Identity(wires=0)


class TestKernel(BaseKernel):
    def __init__(self, device: BaseDevice, feature_map: Callable[[np.ndarray], None]) -> None:
        super().__init__(device, feature_map)

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.dot(x1, x2)


@pytest.fixture(scope="function")
def base_model() -> BaseMLModel:
    kernel = TestKernel(device=DEVICE, feature_map=empty_feature_map)
    return QSVC(kernel=kernel)


@pytest.fixture(scope="function")
def base_experiment(tmp_path: Path) -> Experiment:
    return Experiment(
        name="test_exp",
        desc="test experiment",
        auto_gen_mode=False,
        root_experiment_dirc=tmp_path,
    )


@pytest.fixture(scope="function")
def create_random_dataset() -> Callable:
    def _create_random_dataset(data_num: int, feature_num: int, class_num: int) -> Dataset:
        return Dataset(
            X_train=np.random.rand(data_num, feature_num),
            y_train=np.random.randint(class_num, size=data_num),
            X_val=np.random.rand(data_num, feature_num),
            y_val=np.random.randint(class_num, size=data_num),
            X_test=np.random.rand(data_num, feature_num),
            y_test=np.random.randint(class_num, size=data_num),
            config=DatasetConfig(
                type="generate",
                generate=GenerateDataConfig(generate_method="linear"),
                random_seed=42,
                split=SplitConfig(train_ratio=0.8, validation_ratio=0.0, test_ratio=0.2, shuffle=True),
                features=None,
            ),
        )

    return _create_random_dataset
