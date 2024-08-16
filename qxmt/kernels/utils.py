from pennylane import qml

from qxmt.constants import PENNYLANE_DEVICES
from qxmt.exceptions import InvalidQunatumDeviceError
from qxmt.types import QuantumDeviceType


def get_platform_from_device(device: QuantumDeviceType) -> str:
    """Get the platform name from the device.

    Args:
        device (QuantumDeviceType): quantum device

    Returns:
        str: platform name
    """
    if isinstance(device, PENNYLANE_DEVICES):
        return "pennylane"
    else:
        raise InvalidQunatumDeviceError(f"Device {device} is not supported.")


def get_number_of_qubits(device: QuantumDeviceType) -> int:
    """Get the number of qubits from the device.

    Args:
        device (QuantumDeviceType): quantum device

    Returns:
        int: number of qubits
    """
    if isinstance(device, PENNYLANE_DEVICES):
        return len(device.wires)
    else:
        raise InvalidQunatumDeviceError(f"Device {device} is not supported.")
