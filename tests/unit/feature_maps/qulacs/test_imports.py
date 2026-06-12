import pytest

pytest.importorskip("qulacs")

from qxmt.feature_maps.qulacs import __all__

EXPECTED_ALL = [
    "QulacsBaseFeatureMap",
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
