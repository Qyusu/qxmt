from typing import Any, Callable

import numpy as np
import pennylane as qml
import pytest

from qxmt.devices.base import BaseDevice
from qxmt.kernels import BaseKernel
from qxmt.models import QSVC, QSVR, QRiggeRegressor

DEVICE = BaseDevice(platform="pennylane", name="default.qubit", n_qubits=2, shots=None)


def empty_feature_map(x: np.ndarray) -> None:
    qml.Identity(wires=0)


class TestKernel(BaseKernel):
    def __init__(self, device: BaseDevice, feature_map: Callable[[np.ndarray], None]) -> None:
        super().__init__(device, feature_map)

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray]:
        kernel_value = np.dot(x1, x2)
        probs = np.array([0.2, 0.1, 0.4, 0.0, 0.3])  # dummy probs
        return kernel_value, probs


@pytest.fixture(scope="function")
def build_qsvc() -> Callable:
    def _build_qsvc(**kwargs: Any) -> QSVC:
        kernel = TestKernel(device=DEVICE, feature_map=empty_feature_map)
        return QSVC(kernel=kernel, **kwargs)

    return _build_qsvc


@pytest.fixture(scope="function")
def build_qsvr() -> Callable:
    def _build_qsvr(**kwargs: Any) -> QSVR:
        kernel = TestKernel(device=DEVICE, feature_map=empty_feature_map)
        return QSVR(kernel=kernel, **kwargs)

    return _build_qsvr


@pytest.fixture(scope="function")
def build_qrigge() -> Callable:
    def _build_qsvm(**kwargs: Any) -> QRiggeRegressor:
        kernel = TestKernel(device=DEVICE, feature_map=empty_feature_map)
        return QRiggeRegressor(kernel=kernel, **kwargs)

    return _build_qsvm


# def qsvm_model() -> QSVC:
#     kernel = TestKernel(device=DEVICE, feature_map=empty_feature_map)
#     return QSVC(kernel=kernel)
