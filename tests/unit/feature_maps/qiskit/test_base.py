import numpy as np
import pytest
from pytest_mock import MockerFixture

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit

from qxmt.feature_maps.qiskit.base import QiskitBaseFeatureMap


class TestQiskitBaseFeatureMap:
    @pytest.fixture(scope="function")
    def base_feature_map(self) -> QiskitBaseFeatureMap:
        class ConcreteFeatureMap(QiskitBaseFeatureMap):
            def feature_map(self, x: np.ndarray) -> None:
                self.circuit = QuantumCircuit(self.n_qubits)

        return ConcreteFeatureMap(n_qubits=2)

    def test_init(self, base_feature_map: QiskitBaseFeatureMap) -> None:
        assert base_feature_map.n_qubits == 2
        assert base_feature_map.platform == "qiskit"

    def test_draw(self, base_feature_map: QiskitBaseFeatureMap, mocker: MockerFixture) -> None:
        mock_logger = mocker.Mock()

        base_feature_map.draw(x_dim=2, format="text", logger=mock_logger)

        mock_logger.info.assert_called_once()

    def test_draw_invalid_args(self, base_feature_map: QiskitBaseFeatureMap) -> None:
        with pytest.raises(ValueError, match="Either 'x' or 'x_dim' argument must be provided."):
            base_feature_map.draw()
