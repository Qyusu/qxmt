from qxmt.kernels.pennylane import __all__

EXPECTED_ALL = ["FidelityKernel"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
