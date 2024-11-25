from qxmt.models import __all__

EXPECTED_ALL = [
    "BaseMLModel",
    "BaseKernelModel",
    "ModelBuilder",
    "QSVC",
    "QSVR",
    "QRiggeRegressor",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
