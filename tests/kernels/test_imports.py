from qxmt.kernels import __all__

EXPECTED_ALL = ["BaseKernel"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)