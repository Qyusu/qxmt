from pathlib import Path

import numpy as np
import pennylane as qml
import pytest

from qxmt import Experiment
from qxmt.datasets.schema import Dataset
from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.kernels.base import BaseKernel
from qxmt.models.base import BaseModel
from qxmt.models.qsvm import QSVM

DEVICE = qml.device("default.qubit", wires=2)


class TestKernel(BaseKernel):
    def __init__(self, device: qml.Device) -> None:
        super().__init__(device)

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.dot(x1, x2)


@pytest.fixture(scope="function")
def base_experiment(tmp_path: Path) -> Experiment:
    return Experiment(
        name="test_exp",
        desc="test experiment",
        root_experiment_dirc=tmp_path,
    )


@pytest.fixture(scope="function")
def base_model() -> BaseModel:
    kernel = TestKernel(device=DEVICE)
    return QSVM(kernel=kernel)


@pytest.fixture(scope="function")
def dataset() -> Dataset:
    return Dataset(
        x_train=np.random.rand(10, 10),
        y_train=np.random.randint(2, size=10),
        x_test=np.random.rand(10, 10),
        y_test=np.random.randint(2, size=10),
        features=["feature_1", "feature_2"],
    )
