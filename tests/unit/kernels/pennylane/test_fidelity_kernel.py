import numpy as np
import pennylane as qml
import pytest

from qxmt.devices.base import BaseDevice
from qxmt.kernels.pennylane import FidelityKernel


def empty_feature_map(x: np.ndarray) -> None:
    qml.Identity(wires=0)


@pytest.fixture(scope="function")
def fidelity_kernel() -> FidelityKernel:
    device = BaseDevice(platform="pennylane", device_name="default.qubit", backend_name=None, n_qubits=2, shots=None)
    return FidelityKernel(device, feature_map=empty_feature_map)


class TestFidelityKernel:
    def test_compute_matrix_by_state_vector(self, fidelity_kernel: FidelityKernel) -> None:
        x1 = np.array([[0, 1]])
        x2 = np.array([[1, 0]])
        kernel_matrix = fidelity_kernel._compute_matrix_by_state_vector(x1, x2)
        assert np.array_equal(kernel_matrix, np.array([[1.0]]))

    def test_compute_by_sampling(self, fidelity_kernel: FidelityKernel) -> None:
        pass
