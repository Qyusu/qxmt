from qxmt.datasets.tfds import __all__

EXPECTED_ALL = [
    "TFDSDataLoader",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
