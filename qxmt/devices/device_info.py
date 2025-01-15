from braket.aws import AwsDevice
from pydantic import BaseModel
from qiskit_ibm_runtime import QiskitRuntimeService

from qxmt.constants import PENNYLANE_DEVICES
from qxmt.devices.base import BaseDevice
from qxmt.exceptions import InvalidQunatumDeviceError
from qxmt.types import QuantumDeviceType

STATUS_ONLINE = "ONLINE"
STATUS_OFFLINE = "OFFLINE"


class RemoteDeviceStatus(BaseModel):
    name: str
    n_qubits: int
    status: str


def get_platform_from_device(device: BaseDevice | QuantumDeviceType) -> str:
    """Get the platform name from the device.

    Args:
        device (BaseDevice | QuantumDeviceType): quantum device

    Returns:
        str: platform name
    """
    if isinstance(device, BaseDevice):
        return device.platform

    if isinstance(device, PENNYLANE_DEVICES):
        return "pennylane"
    else:
        raise InvalidQunatumDeviceError(f"Device {device} is not supported.")


def get_number_of_qubits(device: BaseDevice | QuantumDeviceType) -> int:
    """Get the number of qubits from the device.

    Args:
        device (BaseDevice | QuantumDeviceType): quantum device

    Returns:
        int: number of qubits
    """
    if isinstance(device, BaseDevice):
        return device.n_qubits

    if isinstance(device, PENNYLANE_DEVICES):
        return len(device.wires)
    else:
        raise InvalidQunatumDeviceError(f"Device {device} is not supported.")


def get_ibmq_available_devices(service: QiskitRuntimeService) -> list[RemoteDeviceStatus]:
    """Get the available IBMQ devices.
    Each device has the name, number of qubits, and status (Online or Offline).

    Args:
        service (QiskitRuntimeService): authorized IBMQ service

    Returns:
        list[RemoteDeviceStatus]: list of IBMQ devices
    """
    device_list = []
    for backend in service.backends():
        backend_name = backend.name
        qubits = backend.num_qubits
        status = STATUS_ONLINE if backend.status().operational else STATUS_OFFLINE
        device = RemoteDeviceStatus(name=backend_name, n_qubits=qubits, status=status)
        device_list.append(device)

    return device_list


def get_amazon_braket_available_devices() -> list[RemoteDeviceStatus]:
    """Get the available Amazon Braket devices.
    Each device has the name, number of qubits, and status (Online or Offline).

    Returns:
        list[RemoteDeviceStatus]: list of Amazon Braket devices
    """
    device_list = []
    devices = AwsDevice.get_devices()
    for device in devices:
        name = device.name
        n_qubits = device.properties.paradigm.qubitCount  # type: ignore
        status = STATUS_ONLINE if device.status == STATUS_ONLINE else STATUS_OFFLINE
        device_status = RemoteDeviceStatus(name=name, n_qubits=n_qubits, status=status)
        device_list.append(device_status)

    return device_list
