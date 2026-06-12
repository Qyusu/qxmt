import pytest

pytest.importorskip("qiskit")

from qxmt.kernels.qiskit import __all__

EXPECTED_ALL = ["QiskitBaseKernel", "FidelityKernel", "ProjectedKernel"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
