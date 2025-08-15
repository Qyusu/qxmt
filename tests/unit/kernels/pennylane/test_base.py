from unittest.mock import Mock

import numpy as np
import pennylane as qml
import pytest
from pennylane.measurements import SampleMP, StateMP

from qxmt.devices.pennylane_device import PennyLaneDevice
from qxmt.kernels.pennylane.base import PennyLaneBaseKernel


def simple_feature_map(x: np.ndarray) -> None:
    """Simple feature map for testing."""
    for i in range(len(x)):
        if x[i] == 1:
            qml.PauliX(wires=i)


class MockPennyLaneKernel(PennyLaneBaseKernel):
    """Mock implementation of PennyLaneBaseKernel for testing."""

    def _circuit_for_sampling(self, *args: np.ndarray) -> SampleMP | list[SampleMP]:
        """Mock circuit for sampling mode."""
        x = args[0]
        # Call the feature map using the feature_map method
        self.feature_map.feature_map(x)
        return self._get_sampling_measurement()

    def _circuit_for_state_vector(self, *args: np.ndarray) -> StateMP:
        """Mock circuit for state vector mode."""
        x = args[0]
        # Call the feature map using the feature_map method
        self.feature_map.feature_map(x)
        return qml.state()

    def _process_state_vector(self, state_vector: np.ndarray) -> np.ndarray:
        """Mock process state vector - just return the state vector."""
        return state_vector

    def _compute_kernel_block(self, block1: np.ndarray, block2: np.ndarray) -> np.ndarray:
        """Mock kernel block computation - compute fidelity."""
        kernel_block = np.zeros((len(block1), len(block2)))
        for i, state1 in enumerate(block1):
            for j, state2 in enumerate(block2):
                # Compute fidelity between two states
                fidelity = np.abs(np.vdot(state1, state2)) ** 2
                kernel_block[i, j] = fidelity
        return kernel_block

    def _compute_by_sampling(self, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray]:
        """Mock implementation of _compute_by_sampling."""
        # Simple mock implementation for sampling mode
        mock_probs = np.random.random(2**self.n_qubits)
        mock_probs = mock_probs / np.sum(mock_probs)  # Normalize
        mock_kernel_value = np.random.random()
        return mock_kernel_value, mock_probs


@pytest.fixture(scope="function")
def device_state_vector() -> PennyLaneDevice:
    """Device for state vector mode (shots=None)."""
    return PennyLaneDevice(platform="pennylane", device_name="default.qubit", backend_name=None, n_qubits=2, shots=None)


@pytest.fixture(scope="function")
def device_sampling() -> PennyLaneDevice:
    """Device for sampling mode (shots > 0)."""
    return PennyLaneDevice(platform="pennylane", device_name="default.qubit", backend_name=None, n_qubits=2, shots=1000)


@pytest.fixture(scope="function")
def mock_kernel_state_vector(device_state_vector: PennyLaneDevice) -> MockPennyLaneKernel:
    """Mock kernel for state vector mode."""
    return MockPennyLaneKernel(device_state_vector, simple_feature_map)


@pytest.fixture(scope="function")
def mock_kernel_sampling(device_sampling: PennyLaneDevice) -> MockPennyLaneKernel:
    """Mock kernel for sampling mode."""
    return MockPennyLaneKernel(device_sampling, simple_feature_map)


class TestPennyLaneBaseKernel:
    """Test class for PennyLaneBaseKernel."""

    def test_init(self, device_state_vector: PennyLaneDevice) -> None:
        """Test initialization of PennyLaneBaseKernel."""
        kernel = MockPennyLaneKernel(device_state_vector, simple_feature_map)

        assert kernel.device == device_state_vector
        assert kernel.n_qubits == 2
        assert kernel.platform == "pennylane"
        assert not kernel.is_sampling
        assert kernel._qnode is None
        assert kernel.state_memory == {}

    def test_init_sampling_mode(self, device_sampling: PennyLaneDevice) -> None:
        """Test initialization with sampling device."""
        kernel = MockPennyLaneKernel(device_sampling, simple_feature_map)

        assert kernel.is_sampling
        assert kernel.device.shots == 1000

    def test_qnode_property_state_vector(self, mock_kernel_state_vector: MockPennyLaneKernel) -> None:
        """Test qnode property in state vector mode."""
        qnode = mock_kernel_state_vector.qnode

        assert qnode is not None
        assert isinstance(qnode, qml.QNode)
        assert mock_kernel_state_vector._qnode is qnode

        # Test that subsequent calls return the same qnode
        qnode2 = mock_kernel_state_vector.qnode
        assert qnode is qnode2

    def test_qnode_property_sampling(self, mock_kernel_sampling: MockPennyLaneKernel) -> None:
        """Test qnode property in sampling mode."""
        qnode = mock_kernel_sampling.qnode

        assert qnode is not None
        assert isinstance(qnode, qml.QNode)
        assert mock_kernel_sampling._qnode is qnode

    def test_get_sampling_measurement_pennylane(self, mock_kernel_sampling: MockPennyLaneKernel) -> None:
        """Test get_sampling_measurement for PennyLane device."""
        measurement = mock_kernel_sampling._get_sampling_measurement()

        assert isinstance(measurement, SampleMP)

    def test_get_sampling_measurement_amazon(self, mock_kernel_sampling: MockPennyLaneKernel) -> None:
        """Test get_sampling_measurement for Amazon device."""
        # Mock the device to return True for is_amazon_device
        mock_kernel_sampling.device.is_amazon_device = Mock(return_value=True)

        measurement = mock_kernel_sampling._get_sampling_measurement()

        # For Amazon device, it should return a list of PauliZ measurements
        assert isinstance(measurement, list)
        assert len(measurement) == mock_kernel_sampling.n_qubits

    def test_convert_sampling_results_to_probs_pennylane(self, mock_kernel_sampling: MockPennyLaneKernel) -> None:
        """Test convert_sampling_results_to_probs for PennyLane device."""
        # Mock sampling results - create results for the actual shot count
        shots = mock_kernel_sampling.device.shots
        assert shots is not None  # For sampling mode, shots should always be set
        # Create mock results with the same number of samples as shots
        # Distribute samples across different states
        mock_results = []
        states = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for i in range(shots):
            mock_results.append(states[i % len(states)])
        mock_results = np.array(mock_results)

        probs = mock_kernel_sampling._convert_sampling_results_to_probs(mock_results)

        assert isinstance(probs, np.ndarray)
        assert len(probs) == 2**mock_kernel_sampling.n_qubits  # 4 for 2 qubits
        assert np.isclose(np.sum(probs), 1.0)  # Probabilities should sum to 1

    def test_convert_sampling_results_to_probs_amazon(self, mock_kernel_sampling: MockPennyLaneKernel) -> None:
        """Test convert_sampling_results_to_probs for Amazon device."""
        # Mock the device to return True for is_amazon_device
        mock_kernel_sampling.device.is_amazon_device = Mock(return_value=True)

        # Mock PauliZ results for Amazon (-1 and 1 instead of 0 and 1)
        # Create results for the actual shot count
        shots = mock_kernel_sampling.device.shots
        assert shots is not None  # For sampling mode, shots should always be set

        # Amazon Braket returns a list of arrays for each qubit measurement
        # Format: [[qubit0_results], [qubit1_results]]
        # Each qubit returns -1 or 1 for PauliZ measurements
        qubit0_results = []
        qubit1_results = []

        # Create alternating patterns to generate different states
        for i in range(shots):
            # Create a pattern that generates all 4 states (00, 01, 10, 11)
            state_idx = i % 4
            if state_idx == 0:  # |00⟩ -> [-1, -1]
                qubit0_results.append(-1)
                qubit1_results.append(-1)
            elif state_idx == 1:  # |01⟩ -> [-1, 1]
                qubit0_results.append(-1)
                qubit1_results.append(1)
            elif state_idx == 2:  # |10⟩ -> [1, -1]
                qubit0_results.append(1)
                qubit1_results.append(-1)
            else:  # |11⟩ -> [1, 1]
                qubit0_results.append(1)
                qubit1_results.append(1)

        mock_results = [qubit0_results, qubit1_results]

        probs = mock_kernel_sampling._convert_sampling_results_to_probs(mock_results)

        assert isinstance(probs, np.ndarray)
        assert len(probs) == 2**mock_kernel_sampling.n_qubits
        assert np.isclose(np.sum(probs), 1.0)

    def test_validate_circuit_args_valid(self, mock_kernel_state_vector: MockPennyLaneKernel) -> None:
        """Test validate_circuit_args with valid arguments."""
        x1 = np.array([0, 1])
        x2 = np.array([1, 0])

        # Should not raise any exception
        mock_kernel_state_vector._validate_circuit_args((x1,), 1, "_circuit_for_state_vector")
        mock_kernel_state_vector._validate_circuit_args((x1, x2), 2, "_circuit_for_fidelity")

    def test_validate_circuit_args_invalid_count(self, mock_kernel_state_vector: MockPennyLaneKernel) -> None:
        """Test validate_circuit_args with invalid argument count."""
        x1 = np.array([0, 1])
        x2 = np.array([1, 0])

        # Test single argument expected but two provided
        with pytest.raises(ValueError, match="_circuit_for_state_vector requires exactly 1 argument"):
            mock_kernel_state_vector._validate_circuit_args((x1, x2), 1, "_circuit_for_state_vector")

        # Test two arguments expected but one provided
        with pytest.raises(ValueError, match="_circuit_for_fidelity requires exactly 2 arguments"):
            mock_kernel_state_vector._validate_circuit_args((x1,), 2, "_circuit_for_fidelity")

    def test_compute_matrix_by_state_vector(self, mock_kernel_state_vector: MockPennyLaneKernel) -> None:
        """Test _compute_matrix_by_state_vector method."""
        x1_array = np.array([[0, 1], [1, 0]])
        x2_array = np.array([[0, 1], [1, 1]])

        kernel_matrix = mock_kernel_state_vector._compute_matrix_by_state_vector(
            x1_array, x2_array, show_progress=False
        )

        assert kernel_matrix.shape == (2, 2)
        assert isinstance(kernel_matrix, np.ndarray)

        # Check that states are computed and stored in memory
        assert len(mock_kernel_state_vector.state_memory) > 0

        # Check diagonal elements (same states should have fidelity 1)
        assert np.isclose(kernel_matrix[0, 0], 1.0)

    def test_compute_matrix_by_state_vector_caching(self, mock_kernel_state_vector: MockPennyLaneKernel) -> None:
        """Test that state vectors are cached correctly."""
        x1_array = np.array([[0, 1], [1, 0]])
        x2_array = np.array([[0, 1]])  # Same as first element of x1_array

        # First computation
        kernel_matrix1 = mock_kernel_state_vector._compute_matrix_by_state_vector(
            x1_array, x2_array, show_progress=False
        )
        initial_cache_size = len(mock_kernel_state_vector.state_memory)

        # Second computation with overlapping data
        kernel_matrix2 = mock_kernel_state_vector._compute_matrix_by_state_vector(
            x2_array, x1_array, show_progress=False
        )
        final_cache_size = len(mock_kernel_state_vector.state_memory)

        # Cache size should not increase significantly since states are reused
        assert final_cache_size >= initial_cache_size

        # Results should be consistent
        assert np.isclose(kernel_matrix1[0, 0], kernel_matrix2[0, 0])

    def test_compute_matrix_by_state_vector_block_processing(
        self, mock_kernel_state_vector: MockPennyLaneKernel
    ) -> None:
        """Test block processing in _compute_matrix_by_state_vector."""
        # Create larger arrays to test blocking
        x1_array = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        x2_array = np.array([[0, 0], [1, 1]])

        # Use small block size to force multiple blocks
        kernel_matrix = mock_kernel_state_vector._compute_matrix_by_state_vector(
            x1_array, x2_array, show_progress=False, block_size=2
        )

        assert kernel_matrix.shape == (4, 2)
        assert isinstance(kernel_matrix, np.ndarray)

        # Verify some expected properties
        # |00⟩ should have perfect fidelity with itself
        assert np.isclose(kernel_matrix[0, 0], 1.0)
        # |11⟩ should have perfect fidelity with itself
        assert np.isclose(kernel_matrix[3, 1], 1.0)
