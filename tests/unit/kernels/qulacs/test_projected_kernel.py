import numpy as np
import pytest
from pytest_mock import MockerFixture

from qxmt.devices.qulacs_device import QulacsDevice
from qxmt.feature_maps.qulacs.base import QulacsBaseFeatureMap
from qxmt.kernels.qulacs.projected_kernel import ProjectedKernel


class TestProjectedKernel:
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
        return mock_fm

    def test_init(self, device: QulacsDevice, feature_map: QulacsBaseFeatureMap) -> None:
        kernel = ProjectedKernel(device, feature_map, gamma=2.0, projection="z")
        assert isinstance(kernel, ProjectedKernel)
        assert kernel.gamma == 2.0
        assert kernel.projection == "z"

    def test_init_invalid_projection(self, device: QulacsDevice, feature_map: QulacsBaseFeatureMap) -> None:
        with pytest.raises(ValueError, match="Projection method must be 'x', 'y', or 'z'."):
            ProjectedKernel(device, feature_map, projection="invalid")

    def test_apply_projection_gates_x(
        self, device: QulacsDevice, feature_map: QulacsBaseFeatureMap, mocker: MockerFixture
    ) -> None:
        kernel = ProjectedKernel(device, feature_map, projection="x")
        mock_gate = mocker.patch("qxmt.kernels.qulacs.projected_kernel.gate")
        mock_state = mocker.Mock()
        mock_h_gate = mocker.Mock()
        mock_gate.H.return_value = mock_h_gate

        kernel._apply_projection_gates(mock_state)

        assert mock_gate.H.call_count == 2
        assert mock_h_gate.update_quantum_state.call_count == 2
        mock_h_gate.update_quantum_state.assert_called_with(mock_state)

    def test_apply_projection_gates_y(
        self, device: QulacsDevice, feature_map: QulacsBaseFeatureMap, mocker: MockerFixture
    ) -> None:
        kernel = ProjectedKernel(device, feature_map, projection="y")
        mock_gate = mocker.patch("qxmt.kernels.qulacs.projected_kernel.gate")
        mock_state = mocker.Mock()
        mock_ry_gate = mocker.Mock()
        mock_gate.RY.return_value = mock_ry_gate

        kernel._apply_projection_gates(mock_state)

        assert mock_gate.RY.call_count == 2

        mock_gate.RY.assert_any_call(0, np.pi / 2)
        mock_gate.RY.assert_any_call(1, np.pi / 2)
        assert mock_ry_gate.update_quantum_state.call_count == 2

    def test_calculate_expected_values(self, device: QulacsDevice, feature_map: QulacsBaseFeatureMap) -> None:
        kernel = ProjectedKernel(device, feature_map, projection="z")

        probs = np.array([0.5, 0.0, 0.5, 0.0])
        expected = kernel._calculate_expected_values(probs)
        assert np.allclose(expected, np.array([1.0, 0.0]))

    def test_circuit_for_sampling(
        self, device: QulacsDevice, feature_map: QulacsBaseFeatureMap, mocker: MockerFixture
    ) -> None:
        kernel = ProjectedKernel(device, feature_map)
        x = np.array([0.1, 0.2])

        mock_state_cls = mocker.patch("qxmt.kernels.qulacs.projected_kernel.QuantumState")
        mock_state = mock_state_cls.return_value
        mock_state.sampling.return_value = [0, 0, 0]  # 3 samples

        result = kernel._circuit_for_sampling(x)

        mock_state_cls.assert_called_with(2)
        mock_state.set_zero_state.assert_called_once()
        feature_map.circuit.update_quantum_state.assert_called_with(mock_state)
        # projection is Z by default, so _apply_projection_gates does nothing
        assert len(result) == 3

    def test_compute_kernel_block(self, device: QulacsDevice, feature_map: QulacsBaseFeatureMap) -> None:
        kernel = ProjectedKernel(device, feature_map, gamma=1.0)

        b1 = np.array([[1.0, 0.0]])
        b2 = np.array([[0.0, 1.0]])

        res = kernel._compute_kernel_block(b1, b2)
        assert res.shape == (1, 1)
        assert np.isclose(res[0, 0], np.exp(-2.0))

    def test_process_state_vector(self, device: QulacsDevice, feature_map: QulacsBaseFeatureMap) -> None:
        kernel = ProjectedKernel(device, feature_map)

        vec = np.array([1, 0, 0, 0])
        res = kernel._process_state_vector(vec)
        assert np.allclose(res, np.array([1.0, 1.0]))
