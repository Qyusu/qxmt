from qxmt.feature_maps.pennylane import __all__

EXPECTED_ALL = [
    "XXFeatureMap",
    "YYFeatureMap",
    "ZZFeatureMap",
    "HRotationFeatureMap",
    "RotationFeatureMap",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
