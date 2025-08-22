from qxmt.ansatze.pennylane import __all__

EXPECTED_ALL = ["AllSinglesDoublesAnsatz", "GeteFabricAnsatz", "UCCSDAnsatz"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
