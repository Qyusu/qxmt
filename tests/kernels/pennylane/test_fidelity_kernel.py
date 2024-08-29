import numpy as np
import pennylane as qml
import pytest

from qxmt.kernels.pennylane import FidelityKernel


def empty_feature_map(x: np.ndarray) -> None:
    qml.Identity(wires=0)


@pytest.fixture(scope="function")
def fidelity_kernel() -> FidelityKernel:
    device = qml.device("default.qubit", wires=2)
    return FidelityKernel(device, feature_map=empty_feature_map)


class TestFidelityKernel:
    def test_compute(self, fidelity_kernel: FidelityKernel) -> None:
        x1 = np.array([0, 1])
        x2 = np.array([1, 0])
        kernel_value = fidelity_kernel.compute(x1, x2)
        assert kernel_value == 1.0
