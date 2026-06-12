import numpy as np
import pytest
from pytest_mock import MockerFixture

pytest.importorskip("qulacs")

from qulacs import QuantumCircuit

from qxmt.feature_maps.qulacs.base import QulacsBaseFeatureMap


class TestQulacsBaseFeatureMap:
    @pytest.fixture(scope="function")
    def base_feature_map(self) -> QulacsBaseFeatureMap:
        class ConcreteFeatureMap(QulacsBaseFeatureMap):
            def feature_map(self, x: np.ndarray) -> None:
                self.circuit = QuantumCircuit(self.n_qubits)

        return ConcreteFeatureMap(n_qubits=2)

    def test_init(self, base_feature_map: QulacsBaseFeatureMap) -> None:
        assert base_feature_map.n_qubits == 2
        assert base_feature_map.platform == "qulacs"

    def test_draw_latex(self, base_feature_map: QulacsBaseFeatureMap, mocker: MockerFixture) -> None:
        mock_logger = mocker.Mock()
        mock_circuit_drawer = mocker.patch("qxmt.feature_maps.qulacs.base.circuit_drawer")
        mock_circuit_drawer.return_value = "circuit_latex"

        base_feature_map.draw(x_dim=2, format="latex", logger=mock_logger)

        mock_circuit_drawer.assert_called_once()
        mock_logger.info.assert_called_once_with("circuit_latex")

    def test_draw_mpl(self, base_feature_map: QulacsBaseFeatureMap, mocker: MockerFixture) -> None:
        mock_circuit_drawer = mocker.patch("qxmt.feature_maps.qulacs.base.circuit_drawer")
        mock_plt_show = mocker.patch("matplotlib.pyplot.show")

        base_feature_map.draw(x_dim=2, format="mpl")

        mock_circuit_drawer.assert_called_once()
        mock_plt_show.assert_called_once()

    def test_draw_invalid_format(self, base_feature_map: QulacsBaseFeatureMap) -> None:
        with pytest.raises(ValueError, match="Invalid format 'invalid' for drawing the circuit"):
            base_feature_map.draw(x_dim=2, format="invalid")
