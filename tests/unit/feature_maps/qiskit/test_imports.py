import pytest

pytest.importorskip("qiskit")

from qxmt.feature_maps.qiskit import __all__

EXPECTED_ALL = [
    "QiskitBaseFeatureMap",
    "XXFeatureMap",
    "YYFeatureMap",
    "ZZFeatureMap",
    "NPQCFeatureMap",
    "HRotationFeatureMap",
    "RotationFeatureMap",
    "YZCXFeatureMap",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
