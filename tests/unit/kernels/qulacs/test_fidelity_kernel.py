import numpy as np
import pytest
from pytest_mock import MockerFixture

from qxmt.devices.qulacs_device import QulacsDevice
from qxmt.feature_maps.qulacs.base import QulacsBaseFeatureMap
from qxmt.kernels.qulacs.fidelity_kernel import FidelityKernel


class TestFidelityKernel:
    @pytest.fixture(scope="function")
    def device(self) -> QulacsDevice:
        return QulacsDevice(
            platform="qulacs",
            device_name="cpu.simulator",
            backend_name=None,
            n_qubits=2,
            shots=100,
        )

    @pytest.fixture(scope="function")
    def feature_map(self, mocker: MockerFixture) -> QulacsBaseFeatureMap:
        mock_fm = mocker.Mock(spec=QulacsBaseFeatureMap)
        mock_fm.n_qubits = 2
        mock_fm.circuit = mocker.Mock()
        # Mock get_gate_count and get_gate for sampling circuit construction
        mock_fm.circuit.get_gate_count.return_value = 1
        mock_gate = mocker.Mock()
        mock_gate.get_inverse.return_value = mocker.Mock()
        mock_fm.circuit.get_gate.return_value = mock_gate
        return mock_fm

    def test_init(self, device: QulacsDevice, feature_map: QulacsBaseFeatureMap) -> None:
        kernel = FidelityKernel(device, feature_map)
        assert isinstance(kernel, FidelityKernel)

    def test_process_state_vector(self, device: QulacsDevice, feature_map: QulacsBaseFeatureMap) -> None:
        kernel = FidelityKernel(device, feature_map)
        vec = np.array([1, 0, 0, 0])
        assert np.array_equal(kernel._process_state_vector(vec), vec)

    def test_compute_kernel_block(self, device: QulacsDevice, feature_map: QulacsBaseFeatureMap) -> None:
        kernel = FidelityKernel(device, feature_map)
        b1 = np.array([[1, 0], [0, 1]])
        b2 = np.array([[1, 0], [0, 1]])

        block = kernel._compute_kernel_block(b1, b2)
        assert np.array_equal(block, np.eye(2))

    def test_circuit_for_sampling(
        self, device: QulacsDevice, feature_map: QulacsBaseFeatureMap, mocker: MockerFixture
    ) -> None:
        kernel = FidelityKernel(device, feature_map)
        x1 = np.array([0])
        x2 = np.array([1])

        # Mock QuantumState
        mock_state_cls = mocker.patch("qxmt.kernels.qulacs.fidelity_kernel.QuantumState")
        mock_state_instance = mock_state_cls.return_value
        mock_state_instance.sampling.return_value = [0] * 100

        # We need to mock QuantumCircuit created inside the method
        mock_circuit_cls = mocker.patch("qxmt.kernels.qulacs.fidelity_kernel.QuantumCircuit")
        mock_inv_circuit = mock_circuit_cls.return_value

        result = kernel._circuit_for_sampling(x1, x2)

        # Verify inverse circuit was constructed and used
        mock_inv_circuit.add_gate.assert_called()
        mock_inv_circuit.update_quantum_state.assert_called_with(mock_state_instance)

        # Verify sampling
        mock_state_instance.sampling.assert_called_with(100)
        assert len(result) == 100

    def test_compute_by_sampling(
        self, device: QulacsDevice, feature_map: QulacsBaseFeatureMap, mocker: MockerFixture
    ) -> None:
        kernel = FidelityKernel(device, feature_map)

        # Mock _circuit_for_sampling to return all zeros (perfect fidelity state |0>)
        mocker.patch.object(kernel, "_circuit_for_sampling", return_value=[0] * 100)

        val, probs = kernel._compute_by_sampling(np.array([1]), np.array([2]))

        assert val == 1.0
        assert probs[0] == 1.0
