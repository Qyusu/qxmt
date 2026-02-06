from qxmt.kernels.qulacs import __all__

EXPECTED_ALL = ["QulacsBaseKernel", "FidelityKernel", "ProjectedKernel"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
