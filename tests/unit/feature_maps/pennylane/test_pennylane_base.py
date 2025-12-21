import numpy as np
import pennylane as qml
import pytest

from qxmt.exceptions import InputShapeError
from qxmt.feature_maps.pennylane import PennyLaneBaseFeatureMap


class EmptyFeatureMap(PennyLaneBaseFeatureMap):
    def __init__(self, n_qubits: int) -> None:
        super().__init__(n_qubits)

    def feature_map(self, x: np.ndarray) -> None:
        qml.Identity(wires=0)


@pytest.fixture(scope="function")
def base_feature_map() -> PennyLaneBaseFeatureMap:
    return EmptyFeatureMap(n_qubits=2)


class TestPennyLaneBaseFeatureMap:
    def test__init__(self, base_feature_map: PennyLaneBaseFeatureMap) -> None:
        assert base_feature_map.platform == "pennylane"
        assert base_feature_map.n_qubits == 2

    def test_feature_map(self, base_feature_map: PennyLaneBaseFeatureMap) -> None:
        x = np.random.rand(1, 2)
        base_feature_map(x)

    def test_check_input_dim_eq_nqubits(self, base_feature_map: PennyLaneBaseFeatureMap) -> None:
        x = np.random.rand(1, 3)
        with pytest.raises(InputShapeError):
            base_feature_map.check_input_dim_eq_nqubits(x)

    def test_draw(self, base_feature_map: PennyLaneBaseFeatureMap) -> None:
        base_feature_map.draw(x_dim=2)

        with pytest.raises(ValueError):
            base_feature_map.draw()
