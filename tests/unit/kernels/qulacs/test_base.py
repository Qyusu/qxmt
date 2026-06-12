from typing import Any

import numpy as np
import pytest
from pytest_mock import MockerFixture

pytest.importorskip("qulacs")

from qulacs import QuantumState

from qxmt.devices import QulacsDevice
from qxmt.feature_maps import QulacsBaseFeatureMap
from qxmt.kernels import QulacsBaseKernel


class MockQulacsKernel(QulacsBaseKernel):
    """Mock implementation of QulacsBaseKernel for testing."""

    def _circuit_for_sampling(self, *args: np.ndarray) -> np.ndarray:
        return np.array([0])

    def _process_state_vector(self, state_vector: np.ndarray) -> np.ndarray:
        return state_vector

    def _compute_kernel_block(self, block1: np.ndarray, block2: np.ndarray) -> np.ndarray:
        return np.zeros((len(block1), len(block2)))

    def _compute_by_sampling(self, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray]:
        return 0.0, np.array([1.0])


@pytest.fixture(scope="function")
def device() -> QulacsDevice:
    """Qulacs device fixture."""
    return QulacsDevice(
        platform="qulacs",
        device_name="cpu.simulator",
        backend_name=None,
        n_qubits=2,
        shots=1000,
    )


@pytest.fixture(scope="function")
def feature_map(mocker: MockerFixture) -> Any:
    """Mock feature map fixture."""
    mock_fm = mocker.Mock(spec=QulacsBaseFeatureMap)
    mock_fm.n_qubits = 2
    mock_fm.circuit = mocker.Mock()

    def update_state(state: QuantumState) -> None:
        pass

    mock_fm.circuit.update_quantum_state.side_effect = update_state
    return mock_fm


class TestQulacsBaseKernel:
    """Test class for QulacsBaseKernel."""

    def test_init(self, device: QulacsDevice, feature_map: Any) -> None:
        """Test initialization."""
        kernel = MockQulacsKernel(device, feature_map)
        assert kernel.state_memory == {}
        assert isinstance(kernel, QulacsBaseKernel)

    def test_compute_matrix_by_state_vector(
        self, device: QulacsDevice, feature_map: Any, mocker: MockerFixture
    ) -> None:
        """Test state vector matrix computation."""
        kernel = MockQulacsKernel(device, feature_map)

        x1 = np.array([[0], [1]])
        x2 = np.array([[0]])

        mock_state_cls = mocker.patch("qxmt.kernels.qulacs.base.QuantumState")
        mock_state_instance = mock_state_cls.return_value
        mock_state_instance.get_vector.return_value = np.array([1, 0, 0, 0])

        kernel._compute_matrix_by_state_vector(x1, x2, show_progress=False)

        assert feature_map.feature_map.call_count == 2
        assert feature_map.circuit.update_quantum_state.call_count == 2
        assert (0,) in kernel.state_memory
        assert (1,) in kernel.state_memory

    def test_convert_sampling_results_to_probs(self, device: QulacsDevice, feature_map: Any) -> None:
        """Test conversion of sampling results."""
        device.shots = 10
        kernel = MockQulacsKernel(device, feature_map)

        results = [0, 0, 0, 1, 1, 2, 3, 3, 3, 3]
        probs = kernel._convert_sampling_results_to_probs(results)

        assert len(probs) == 4
        assert probs[0] == 0.3
        assert probs[1] == 0.2
        assert probs[2] == 0.1
        assert probs[3] == 0.4

    def test_validate_circuit_args(self, device: QulacsDevice, feature_map: Any) -> None:
        """Test argument validation."""
        kernel = MockQulacsKernel(device, feature_map)

        with pytest.raises(ValueError):
            kernel._validate_circuit_args((np.array([1]),), 2, "func")

        with pytest.raises(ValueError):
            kernel._validate_circuit_args((np.array([1]), np.array([2])), 1, "func")

        kernel._validate_circuit_args((np.array([1]),), 1, "func")
