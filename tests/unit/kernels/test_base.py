from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from qxmt.devices.base import BaseDevice
from qxmt.devices.pennylane_device import PennyLaneDevice
from qxmt.exceptions import DeviceSettingError
from qxmt.kernels import BaseKernel

N_QUBITS = 2
SHOTS = 5
STATE_VECTOR_DEVICE = PennyLaneDevice(
    platform="pennylane", device_name="default.qubit", backend_name=None, n_qubits=N_QUBITS, shots=None
)
SAMPLING_DEVICE = PennyLaneDevice(
    platform="pennylane", device_name="default.qubit", backend_name=None, n_qubits=N_QUBITS, shots=SHOTS
)
MULTI_QUBITS_DEVICE = PennyLaneDevice(
    platform="pennylane", device_name="default.qubit", backend_name=None, n_qubits=5, shots=SHOTS
)


def empty_feature_map(x: np.ndarray) -> None:
    pass


def invalid_feature_map(x: np.ndarray) -> None:
    raise ValueError("invalid feature map")


class SimpleKernel(BaseKernel):
    def __init__(self, device: BaseDevice, feature_map: Callable[[np.ndarray], None]) -> None:
        super().__init__(device, feature_map)

    def _compute_matrix_by_state_vector(
        self, x1_array: np.ndarray, x2_array: np.ndarray, bar_label: str = "", progress: bool = True
    ) -> np.ndarray:
        self.feature_map(x1_array[0])
        kernel_matrix = np.zeros((len(x1_array), len(x2_array)))
        for i, x1 in enumerate(x1_array):
            for j, x2 in enumerate(x2_array):
                kernel_matrix[i, j] = np.dot(x1, x2)
        return kernel_matrix

    def _compute_by_sampling(self, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray]:
        self.feature_map(x1)
        kernel_value = np.dot(x1, x2)
        probs = np.array([0.2, 0.1, 0.4, 0.0, 0.3])  # dummy probs
        return kernel_value, probs


@pytest.fixture(scope="function")
def kernel_by_state_vector() -> BaseKernel:
    return SimpleKernel(device=STATE_VECTOR_DEVICE, feature_map=empty_feature_map)


@pytest.fixture(scope="function")
def invalid_kernel() -> BaseKernel:
    return SimpleKernel(device=STATE_VECTOR_DEVICE, feature_map=invalid_feature_map)


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

    def test_compute_matrix(self, kernel_by_state_vector: BaseKernel, invalid_kernel: BaseKernel) -> None:
        x_array_1 = np.array([[0, 1], [1, 0]])
        x_array_2 = np.array([[1, 0], [1, 0]])

        kernel_matrix, shots_matrix = kernel_by_state_vector.compute_matrix(
            x_array_1, x_array_2, return_shots_resutls=False, n_jobs=1
        )
        assert kernel_matrix.shape == (2, 2)
        assert shots_matrix is None
        assert np.array_equal(kernel_matrix, np.array([[0.0, 0.0], [1.0, 1.0]]))

        # state vector mode can't get shots results
        # so, return_shots_resutls=True but return None
        kernel_matrix, shots_matrix = kernel_by_state_vector.compute_matrix(
            x_array_1, x_array_2, return_shots_resutls=True, n_jobs=1
        )
        assert shots_matrix is None

        # raise error in compute method
        with pytest.raises(ValueError):
            invalid_kernel.compute_matrix(x_array_1, x_array_2)

    def test_save_shots_results(
        self, kernel_by_state_vector: BaseKernel, kernel_by_sampling: BaseKernel, tmp_path: Path
    ) -> None:
        prob_matrix = np.zeros((10, 10, 4))  # dummy probs matrix, sample=10, n_qubits=2
        valid_save_path = tmp_path / "shots.h5"
        # state vector mode can't save shots results
        with pytest.raises(DeviceSettingError):
            kernel_by_state_vector.save_shots_results(prob_matrix, valid_save_path)

        # invalid extension
        invalid_save_path = tmp_path / "shots.csv"
        with pytest.raises(ValueError):
            kernel_by_sampling.save_shots_results(prob_matrix, invalid_save_path)

        # sampling mode can save shots results
        kernel_by_sampling.save_shots_results(prob_matrix, valid_save_path)
        assert valid_save_path.exists()
