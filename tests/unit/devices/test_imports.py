import subprocess
import sys

from qxmt.devices import __all__

EXPECTED_ALL = [
    "AmazonBraketDevice",
    "BaseDevice",
    "DeviceBuilder",
    "get_number_of_qubits",
    "get_platform_from_device",
    "IBMQDevice",
    "PennyLaneDevice",
    "QiskitDevice",
    "QulacsDevice",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)


def test_devices_import_does_not_load_optional_remote_dependencies() -> None:
    code = (
        "import sys; "
        "import qxmt.devices; "
        "print('braket' in sys.modules); "
        "print('qiskit' in sys.modules); "
        "print('qiskit_ibm_runtime' in sys.modules)"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == ["False", "False", "False"]
