from typing import Callable

import numpy as np
import pytest

from qxmt.devices.base import BaseDevice
from qxmt.exceptions import DeviceSettingError
from qxmt.kernels import BaseKernel

N_QUBITS = 2
SHOTS = 5
STATE_VECTOR_DEVICE = BaseDevice(platform="pennylane", name="default.qubit", n_qubits=N_QUBITS, shots=None)
SAMPLING_DEVICE = BaseDevice(platform="pennylane", name="default.qubit", n_qubits=N_QUBITS, shots=SHOTS)
MULTI_QUBITS_DEVICE = BaseDevice(platform="pennylane", name="default.qubit", n_qubits=5, shots=SHOTS)


def empty_feature_map(x: np.ndarray) -> None:
    pass


class SimpleKernel(BaseKernel):
    def __init__(self, device: BaseDevice, feature_map: Callable[[np.ndarray], None]) -> None:
        super().__init__(device, feature_map)

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.dot(x1, x2)


@pytest.fixture(scope="function")
def kernel_by_state_vector() -> BaseKernel:
    return SimpleKernel(device=STATE_VECTOR_DEVICE, feature_map=empty_feature_map)


@pytest.fixture(scope="function")
def kernel_by_sampling() -> BaseKernel:
    return SimpleKernel(device=SAMPLING_DEVICE, feature_map=empty_feature_map)


@pytest.fixture(scope="function")
def kernel_by_multi_qubits() -> BaseKernel:
    return SimpleKernel(device=MULTI_QUBITS_DEVICE, feature_map=empty_feature_map)


class TestBaseKernel:
    def test__init__(self, kernel_by_state_vector: BaseKernel, kernel_by_sampling: BaseKernel) -> None:
        assert kernel_by_state_vector.device == STATE_VECTOR_DEVICE
        assert kernel_by_state_vector.n_qubits == N_QUBITS
        assert kernel_by_state_vector.platform == "pennylane"
        assert kernel_by_state_vector.feature_map is not None
        assert kernel_by_state_vector.is_sampling is False

        assert kernel_by_sampling.device == SAMPLING_DEVICE
        assert kernel_by_sampling.n_qubits == N_QUBITS
        assert kernel_by_sampling.platform == "pennylane"
        assert kernel_by_sampling.feature_map is not None
        assert kernel_by_sampling.is_sampling is True

    def test__validate_sampling_values(
        self, kernel_by_state_vector: BaseKernel, kernel_by_sampling: BaseKernel
    ) -> None:
        sampling_result = np.array([[0, 1], [1, 0], [0, 0], [1, 1], [0, 1]])

        # valid pattern
        kernel_by_sampling._validate_sampling_values(sampling_result, valid_values=[0, 1])

        # invalid pattern
        # raise error if not sampling mode device
        with pytest.raises(DeviceSettingError):
            kernel_by_state_vector._validate_sampling_values(sampling_result)

        # raise error if invalid values
        with pytest.raises(ValueError):
            kernel_by_sampling._validate_sampling_values(sampling_result, valid_values=[-1, 1])

    def test__generate_all_observable_states(
        self, kernel_by_sampling: BaseKernel, kernel_by_multi_qubits: BaseKernel
    ) -> None:
        state_pattern = "01"
        observable_states = kernel_by_sampling._generate_all_observable_states(state_pattern)
        assert observable_states == ["00", "01", "10", "11"]

        observable_states = kernel_by_multi_qubits._generate_all_observable_states(state_pattern)
        # fmt: off
        assert observable_states == [
            "00000", "00001", "00010", "00011", "00100", "00101", "00110", "00111",
            "01000", "01001", "01010", "01011", "01100", "01101", "01110", "01111",
            "10000", "10001", "10010", "10011", "10100", "10101", "10110", "10111",
            "11000", "11001", "11010", "11011", "11100", "11101", "11110", "11111"
        ]
        # fmt: on

    def test_compute(self, kernel_by_state_vector: BaseKernel) -> None:
        x1 = np.array([0, 1])
        x2 = np.array([1, 0])
        assert kernel_by_state_vector.compute(x1, x2) == 0.0

        x1 = np.array([1, 0])
        x2 = np.array([1, 0])
        assert kernel_by_state_vector.compute(x1, x2) == 1.0

    def test_compute_matrix(self, kernel_by_state_vector: BaseKernel) -> None:
        x_array_1 = np.array([[0, 1], [1, 0]])
        x_array_2 = np.array([[1, 0], [1, 0]])
        kernel_matrix = kernel_by_state_vector.compute_matrix(x_array_1, x_array_2)
        assert kernel_matrix.shape == (2, 2)
        assert np.array_equal(kernel_matrix, np.array([[0.0, 0.0], [1.0, 1.0]]))
