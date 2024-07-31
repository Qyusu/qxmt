from typing import Callable

import numpy as np
import pennylane as qml
import pytest

from qxmt.kernels.base import BaseKernel
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
def qsvm_model() -> QSVM:
    kernel = TestKernel(device=DEVICE, feature_map=empty_feature_map)
    return QSVM(kernel=kernel)
