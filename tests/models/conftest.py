from typing import Any, Callable

import numpy as np
import pennylane as qml
import pytest

from qxmt.devices.base import BaseDevice
from qxmt.kernels import BaseKernel
from qxmt.models import QSVM

DEVICE = BaseDevice(platform="pennylane", name="default.qubit", n_qubits=2, shots=None)


def empty_feature_map(x: np.ndarray) -> None:
    qml.Identity(wires=0)


class TestKernel(BaseKernel):
    def __init__(self, device: BaseDevice, feature_map: Callable[[np.ndarray], None]) -> None:
        super().__init__(device, feature_map)

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.dot(x1, x2)


@pytest.fixture(scope="function")
def build_qsvm() -> Callable:
    def _build_qsvm(**kwargs: Any) -> QSVM:
        kernel = TestKernel(device=DEVICE, feature_map=empty_feature_map)
        return QSVM(kernel=kernel, **kwargs)

    return _build_qsvm


# def qsvm_model() -> QSVM:
#     kernel = TestKernel(device=DEVICE, feature_map=empty_feature_map)
#     return QSVM(kernel=kernel)
