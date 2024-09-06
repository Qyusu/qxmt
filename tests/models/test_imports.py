from qxmt.models import __all__

EXPECTED_ALL = [
    "BaseMLModel",
    "BaseKernelModel",
    "ModelBuilder",
    "QSVM",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
