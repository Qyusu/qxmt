from qxmt.models import __all__

EXPECTED_ALL = ["ModelBuilder"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
