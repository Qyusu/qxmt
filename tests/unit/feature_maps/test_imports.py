from qxmt.feature_maps import __all__

EXPECTED_ALL = ["BaseFeatureMap"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
