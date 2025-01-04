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

DEVICE_STATEVC = BaseDevice(
    platform="pennylane", device_name="default.qubit", backend_name=None, n_qubits=2, shots=None
)
DEVICE_SHOTS = BaseDevice(platform="pennylane", device_name="default.qubit", backend_name=None, n_qubits=2, shots=5)


def empty_feature_map(x: np.ndarray) -> None:
    qml.Identity(wires=0)


class TestKernel(BaseKernel):
    def __init__(self, device: BaseDevice, feature_map: Callable[[np.ndarray], None]) -> None:
        super().__init__(device, feature_map)

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray]:
        kernel_value = np.dot(x1, x2)
        probs = np.array([0.2, 0.1, 0.4, 0.3])  # dummy probs
        return kernel_value, probs


@pytest.fixture(scope="function")
def state_vec_model() -> BaseMLModel:
    kernel = TestKernel(device=DEVICE_STATEVC, feature_map=empty_feature_map)
    return QSVC(kernel=kernel, n_jobs=1)


@pytest.fixture(scope="function")
def shots_model() -> BaseMLModel:
    kernel = TestKernel(device=DEVICE_SHOTS, feature_map=empty_feature_map)
    return QSVC(kernel=kernel, n_jobs=1)


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
                generate=GenerateDataConfig(generate_method="linear"),
                random_seed=42,
                split=SplitConfig(train_ratio=0.8, validation_ratio=0.0, test_ratio=0.2, shuffle=True),
                features=None,
            ),
        )

    return _create_random_dataset
