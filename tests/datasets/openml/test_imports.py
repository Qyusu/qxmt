from qxmt.datasets.openml import __all__

EXPECTED_ALL = [
    "OpenMLDataLoader",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
