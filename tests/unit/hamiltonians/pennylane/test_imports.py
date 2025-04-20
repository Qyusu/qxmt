from qxmt.hamiltonians.pennylane import __all__

EXPECTED_ALL = ["MolecularHamiltonian"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
