import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qiskit import QuantumCircuit

from qxmt.devices.qiskit_device import QiskitDevice
from qxmt.feature_maps.qiskit.rotation import RotationFeatureMap
from qxmt.kernels.qiskit.fidelity_kernel import FidelityKernel


@pytest.fixture(scope="function")
def statevector_device() -> QiskitDevice:
    return QiskitDevice(
        platform="qiskit",
        device_name="statevector",
        backend_name=None,
        n_qubits=2,
        shots=None,
        device_options=None,
    )


@pytest.fixture(scope="function")
def sampling_device() -> QiskitDevice:
    return QiskitDevice(
        platform="qiskit",
        device_name="automatic",
        backend_name=None,
        n_qubits=2,
        shots=100,
        device_options={"seed_simulator": 42},
    )


class TestFidelityKernel:
    def test_process_state_vector(self, statevector_device: QiskitDevice) -> None:
        kernel = FidelityKernel(statevector_device, RotationFeatureMap(2, 1, ["X"]))
        vec = np.array([1, 0, 0, 0])
        assert np.array_equal(kernel._process_state_vector(vec), vec)

    def test_compute_kernel_block(self, statevector_device: QiskitDevice) -> None:
        kernel = FidelityKernel(statevector_device, RotationFeatureMap(2, 1, ["X"]))
        b1 = np.array([[1, 0], [0, 1]])
        b2 = np.array([[1, 0], [0, 1]])

        block = kernel._compute_kernel_block(b1, b2)

        assert np.array_equal(block, np.eye(2))

    def test_circuit_for_sampling(self, sampling_device: QiskitDevice) -> None:
        kernel = FidelityKernel(sampling_device, RotationFeatureMap(2, 1, ["X"]))
        circuit = kernel._circuit_for_sampling(np.array([0.1, 0.2]), np.array([0.1, 0.2]))

        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == 2
        assert circuit.size() > 0

    def test_compute_by_sampling(self, sampling_device: QiskitDevice) -> None:
        kernel = FidelityKernel(sampling_device, RotationFeatureMap(2, 1, ["X"]))

        val, probs = kernel._compute_by_sampling(np.array([0.1, 0.2]), np.array([0.1, 0.2]))

        assert val == 1.0
        assert probs[0] == 1.0

    def test_compute_matrix_by_state_vector(self, statevector_device: QiskitDevice) -> None:
        kernel = FidelityKernel(statevector_device, RotationFeatureMap(2, 1, ["X"]))
        x = np.array([[0.1, 0.2], [0.1, 0.2]])

        matrix, shots = kernel.compute_matrix(x, x, show_progress=False)

        assert shots is None
        assert np.allclose(matrix, np.ones((2, 2)))
