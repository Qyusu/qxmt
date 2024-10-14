from qxmt.datasets.dummy import __all__

EXPECTED_ALL = [
    "generate_linear_separable_data",
    "generate_linear_regression_data",
    "load_dummy_dataset",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
