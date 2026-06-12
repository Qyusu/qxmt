import numpy as np
import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit

from qxmt.devices.qiskit_device import QiskitDevice
from qxmt.feature_maps.qiskit.base import QiskitBaseFeatureMap
from qxmt.kernels.qiskit.base import QiskitBaseKernel


class SimpleFeatureMap(QiskitBaseFeatureMap):
    def feature_map(self, x: np.ndarray) -> None:
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.rx(float(x[0]), 0)


class MockQiskitKernel(QiskitBaseKernel):
    def _circuit_for_sampling(self, *args: np.ndarray) -> QuantumCircuit:
        return QuantumCircuit(self.n_qubits)

    def _process_state_vector(self, state_vector: np.ndarray) -> np.ndarray:
        return state_vector

    def _compute_kernel_block(self, block1: np.ndarray, block2: np.ndarray) -> np.ndarray:
        return np.zeros((len(block1), len(block2)))

    def _compute_by_sampling(self, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray]:
        return 0.0, np.array([1.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope="function")
def device() -> QiskitDevice:
    return QiskitDevice(
        platform="qiskit",
        device_name="statevector",
        backend_name=None,
        n_qubits=2,
        shots=None,
        device_options=None,
    )


class TestQiskitBaseKernel:
    def test_init(self, device: QiskitDevice) -> None:
        kernel = MockQiskitKernel(device, SimpleFeatureMap(n_qubits=2))
        assert kernel.state_memory == {}
        assert isinstance(kernel, QiskitBaseKernel)

    def test_compute_matrix_by_state_vector(self, device: QiskitDevice) -> None:
        kernel = MockQiskitKernel(device, SimpleFeatureMap(n_qubits=2))
        x1 = np.array([[0.0], [1.0]])
        x2 = np.array([[0.0]])

        kernel._compute_matrix_by_state_vector(x1, x2, show_progress=False)

        assert (0.0,) in kernel.state_memory
        assert (1.0,) in kernel.state_memory

    def test_convert_counts_to_probs(self, device: QiskitDevice) -> None:
        device.shots = 10
        kernel = MockQiskitKernel(device, SimpleFeatureMap(n_qubits=2))

        probs = kernel._convert_counts_to_probs({"00": 3, "01": 2, "10": 1, "11": 4})

        assert np.allclose(probs, np.array([0.3, 0.2, 0.1, 0.4]))

    def test_validate_circuit_args(self, device: QiskitDevice) -> None:
        kernel = MockQiskitKernel(device, SimpleFeatureMap(n_qubits=2))

        with pytest.raises(ValueError):
            kernel._validate_circuit_args((np.array([1]),), 2, "func")

        with pytest.raises(ValueError):
            kernel._validate_circuit_args((np.array([1]), np.array([2])), 1, "func")

        kernel._validate_circuit_args((np.array([1]),), 1, "func")
