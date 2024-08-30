import pennylane as qml
import pytest

from qxmt.exceptions import InvalidQunatumDeviceError
from qxmt.utils import get_number_of_qubits, get_platform_from_device


class TestGetPlatformFromDevice:
    def test_get_platform_from_device_pennylane(self) -> None:
        qml_device_list = [
            qml.device("default.qubit", wires=1),
            qml.device("default.mixed", wires=1),
            qml.device("lightning.qubit", wires=1),
            qml.device("default.qutrit", wires=1),
            qml.device("default.qutrit.mixed", wires=1),
            qml.device("default.gaussian", wires=1),
            # qml.device("default.clifford", wires=1), # need to install stim
            # qml.device("default.tensor", wires=1),  # need to install quimb
        ]

        for device in qml_device_list:
            platform = get_platform_from_device(device)
            assert platform == "pennylane"

    def test_get_platform_from_device_not_supported(self) -> None:
        device = "unsupported_device"
        with pytest.raises(InvalidQunatumDeviceError):
            get_platform_from_device(device)


class TestGetNumberOfQubits:
    def test_get_number_of_qubits_pennylane(self) -> None:
        device = qml.device("default.qubit", wires=3)
        n_qubits = get_number_of_qubits(device)
        assert n_qubits == 3

    def test_get_number_of_qubits_not_supported(self) -> None:
        device = "unsupported_device"
        with pytest.raises(InvalidQunatumDeviceError):
            get_number_of_qubits(device)
