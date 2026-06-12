import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qiskit import QuantumCircuit

from qxmt.devices.qiskit_device import QiskitDevice
from qxmt.feature_maps.qiskit.rotation import RotationFeatureMap
from qxmt.kernels.qiskit.projected_kernel import ProjectedKernel


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


class TestProjectedKernel:
    def test_init(self, statevector_device: QiskitDevice) -> None:
        kernel = ProjectedKernel(statevector_device, RotationFeatureMap(2, 1, ["X"]), gamma=2.0, projection="z")
        assert isinstance(kernel, ProjectedKernel)
        assert kernel.gamma == 2.0
        assert kernel.projection == "z"

    def test_init_invalid_projection(self, statevector_device: QiskitDevice) -> None:
        with pytest.raises(ValueError, match="Projection method must be 'x', 'y', or 'z'."):
            ProjectedKernel(statevector_device, RotationFeatureMap(2, 1, ["X"]), projection="invalid")  # type: ignore[arg-type]

    def test_apply_projection_gates_x(self, statevector_device: QiskitDevice) -> None:
        kernel = ProjectedKernel(statevector_device, RotationFeatureMap(2, 1, ["X"]), projection="x")
        circuit = QuantumCircuit(2)

        kernel._apply_projection_gates(circuit)

        assert [instruction.operation.name for instruction in circuit.data] == ["h", "h"]

    def test_apply_projection_gates_y(self, statevector_device: QiskitDevice) -> None:
        kernel = ProjectedKernel(statevector_device, RotationFeatureMap(2, 1, ["X"]), projection="y")
        circuit = QuantumCircuit(2)

        kernel._apply_projection_gates(circuit)

        assert [instruction.operation.name for instruction in circuit.data] == ["ry", "ry"]

    def test_calculate_expected_values(self, statevector_device: QiskitDevice) -> None:
        kernel = ProjectedKernel(statevector_device, RotationFeatureMap(2, 1, ["X"]), projection="z")

        probs = np.array([0.5, 0.0, 0.5, 0.0])
        expected = kernel._calculate_expected_values(probs)

        assert np.allclose(expected, np.array([1.0, 0.0]))

    def test_circuit_for_sampling(self, sampling_device: QiskitDevice) -> None:
        kernel = ProjectedKernel(sampling_device, RotationFeatureMap(2, 1, ["X"]), projection="x")
        circuit = kernel._circuit_for_sampling(np.array([0.1, 0.2]))

        assert isinstance(circuit, QuantumCircuit)
        assert "h" in [instruction.operation.name for instruction in circuit.data]

    def test_compute_kernel_block(self, statevector_device: QiskitDevice) -> None:
        kernel = ProjectedKernel(statevector_device, RotationFeatureMap(2, 1, ["X"]), gamma=1.0)
        b1 = np.array([[1.0, 0.0]])
        b2 = np.array([[0.0, 1.0]])

        res = kernel._compute_kernel_block(b1, b2)

        assert res.shape == (1, 1)
        assert np.isclose(res[0, 0], np.exp(-2.0))

    def test_process_state_vector(self, statevector_device: QiskitDevice) -> None:
        kernel = ProjectedKernel(statevector_device, RotationFeatureMap(2, 1, ["X"]))

        vec = np.array([1, 0, 0, 0])
        res = kernel._process_state_vector(vec)

        assert np.allclose(res, np.array([1.0, 1.0]))

    def test_compute_by_sampling(self, sampling_device: QiskitDevice) -> None:
        kernel = ProjectedKernel(sampling_device, RotationFeatureMap(2, 1, ["X"]))

        val, probs = kernel._compute_by_sampling(np.array([0.1, 0.2]), np.array([0.1, 0.2]))

        assert np.isclose(val, 1.0)
        assert np.isclose(np.sum(probs), 1.0)
