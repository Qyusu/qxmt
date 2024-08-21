from qxmt.models import __all__

EXPECTED_ALL = [
    "BaseModel",
    "BaseKernelModel",
    "ModelBuilder",
    "QSVM",
    "DeviceConfig",
    "ModelConfig",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
