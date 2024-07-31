import numpy as np
import pennylane as qml
import pytest

from qxmt.exceptions import InputShapeError
from qxmt.feature_maps.base import BaseFeatureMap


@pytest.fixture(scope="function")
def base_feature_map() -> BaseFeatureMap:
    class EmptyFeatureMap(BaseFeatureMap):
        def __init__(self, n_qubits: int) -> None:
            super().__init__(n_qubits)

        def feature_map(self, x: np.ndarray) -> None:
            qml.Identity(wires=0)

    return EmptyFeatureMap(n_qubits=2)


class TestBaseFeatureMap:
    def test__init__(self, base_feature_map: BaseFeatureMap) -> None:
        assert base_feature_map.n_qubits == 2

    def test_feature_map(self, base_feature_map: BaseFeatureMap) -> None:
        x = np.random.rand(1, 2)
        base_feature_map(x)

    def test_check_input_shape(self, base_feature_map: BaseFeatureMap) -> None:
        x = np.random.rand(1, 3)
        with pytest.raises(InputShapeError):
            base_feature_map.check_input_shape(x)

    def test_print_circuit(self, base_feature_map: BaseFeatureMap) -> None:
        base_feature_map.print_circuit()
