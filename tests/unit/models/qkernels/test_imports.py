from qxmt.models.qkernels import __all__

EXPECTED_ALL = [
    "BaseMLModel",
    "BaseKernelModel",
    "KernelModelBuilder",
    "QSVC",
    "QSVR",
    "QRiggeRegressor",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
