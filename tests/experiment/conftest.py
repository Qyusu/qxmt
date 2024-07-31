from pathlib import Path
from typing import Callable

import numpy as np
import pennylane as qml
import pytest

from qxmt import Experiment
from qxmt.datasets.schema import Dataset
from qxmt.kernels.base import BaseKernel
from qxmt.models.base import BaseModel
from qxmt.models.qsvm import QSVM

DEVICE = qml.device("default.qubit", wires=2)


def empty_feature_map(x: np.ndarray) -> None:
    qml.Identity(wires=0)


class TestKernel(BaseKernel):
    def __init__(self, device: qml.Device, feature_map: Callable[[np.ndarray], None]) -> None:
        super().__init__(device, feature_map)

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.dot(x1, x2)


@pytest.fixture(scope="function")
def base_model() -> BaseModel:
    kernel = TestKernel(device=DEVICE, feature_map=empty_feature_map)
    return QSVM(kernel=kernel)


@pytest.fixture(scope="function")
def base_experiment(tmp_path: Path) -> Experiment:
    return Experiment(
        name="test_exp",
        desc="test experiment",
        root_experiment_dirc=tmp_path,
    )


@pytest.fixture(scope="function")
def create_random_dataset() -> Callable:
    def _create_random_dataset(data_num: int, feature_num: int, class_num: int) -> Dataset:
        return Dataset(
            x_train=np.random.rand(data_num, feature_num),
            y_train=np.random.randint(class_num, size=data_num),
            x_test=np.random.rand(data_num, feature_num),
            y_test=np.random.randint(class_num, size=data_num),
            features=[f"feature_{i+1}" for i in range(feature_num)],
        )

    return _create_random_dataset
