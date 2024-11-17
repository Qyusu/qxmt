import numpy as np
import pennylane as qml
import pytest

from qxmt.exceptions import InputShapeError
from qxmt.feature_maps import BaseFeatureMap


class EmptyFeatureMap(BaseFeatureMap):
    def __init__(self, platform: str, n_qubits: int) -> None:
        super().__init__(platform, n_qubits)

    def feature_map(self, x: np.ndarray) -> None:
        qml.Identity(wires=0)


@pytest.fixture(scope="function")
def base_feature_map() -> BaseFeatureMap:
    return EmptyFeatureMap(platform="pennylane", n_qubits=2)


class TestBaseFeatureMap:
    def test__init__(self, base_feature_map: BaseFeatureMap) -> None:
        assert base_feature_map.platform == "pennylane"
        assert base_feature_map.n_qubits == 2

    def test_feature_map(self, base_feature_map: BaseFeatureMap) -> None:
        x = np.random.rand(1, 2)
        base_feature_map(x)

    def test_check_input_dim_eq_nqubits(self, base_feature_map: BaseFeatureMap) -> None:
        x = np.random.rand(1, 3)
        with pytest.raises(InputShapeError):
            base_feature_map.check_input_dim_eq_nqubits(x)

    def test_draw(self, base_feature_map: BaseFeatureMap) -> None:
        base_feature_map.draw(x_dim=2)

        with pytest.raises(ValueError):
            base_feature_map.draw()

        with pytest.raises(NotImplementedError):
            base_feature_map.platform = "unsupported"
            base_feature_map.draw(x_dim=2)
