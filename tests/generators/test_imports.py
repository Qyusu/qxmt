from qxmt.generators import __all__

EXPECTED_ALL = [
    "DescriptionGenerator",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
