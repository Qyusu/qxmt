from qxmt.datasets import __all__

EXPECTED_ALL = [
    "Dataset",
    "DatasetConfig",
    "PathConfig",
    "DatasetBuilder",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
