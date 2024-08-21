from typing import Callable

import numpy as np
import pennylane as qml
import pytest

from qxmt.kernels.base import BaseKernel

N_QUBITS = 2
DEVICE = qml.device("default.qubit", wires=N_QUBITS)


def empty_feature_map(x: np.ndarray) -> None:
    pass


class TestKernel(BaseKernel):
    def __init__(self, device: qml.Device, feature_map: Callable[[np.ndarray], None]) -> None:
        super().__init__(device, feature_map)

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.dot(x1, x2)


@pytest.fixture(scope="function")
def base_kernel() -> BaseKernel:
    return TestKernel(device=DEVICE, feature_map=empty_feature_map)


class TestBaseKernel:
    def test__init__(self, base_kernel: BaseKernel) -> None:
        assert base_kernel.device == DEVICE
        assert base_kernel.n_qubits == N_QUBITS
        assert base_kernel.platform == "pennylane"
        assert base_kernel.feature_map is not None

    def test_compute(self, base_kernel: BaseKernel) -> None:
        x1 = np.array([0, 1])
        x2 = np.array([1, 0])
        assert base_kernel.compute(x1, x2) == 0.0

        x1 = np.array([1, 0])
        x2 = np.array([1, 0])
        assert base_kernel.compute(x1, x2) == 1.0

    def test_compute_matrix(self, base_kernel: BaseKernel) -> None:
        x_array_1 = np.array([[0, 1], [1, 0]])
        x_array_2 = np.array([[1, 0], [1, 0]])
        kernel_matrix = base_kernel.compute_matrix(x_array_1, x_array_2)
        assert kernel_matrix.shape == (2, 2)
        assert np.array_equal(kernel_matrix, np.array([[0.0, 0.0], [1.0, 1.0]]))
