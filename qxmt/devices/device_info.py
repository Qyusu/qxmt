from pydantic import BaseModel
from typing import Any

from qxmt.constants import PENNYLANE_DEVICES, PENNYLANE_PLATFORM
from qxmt.devices.base import BaseDevice
from qxmt.exceptions import AmazonBraketSettingError, InvalidQunatumDeviceError

STATUS_ONLINE = "ONLINE"
STATUS_OFFLINE = "OFFLINE"


class RemoteDeviceStatus(BaseModel):
    name: str
    n_qubits: int
    status: str


def get_platform_from_device(device: BaseDevice | object) -> str:
    """Get the platform name from the device.

    Args:
        device (BaseDevice | object): quantum device

    Returns:
        str: platform name
    """
    if isinstance(device, BaseDevice):
        return device.platform

    if isinstance(device, PENNYLANE_DEVICES):
        return PENNYLANE_PLATFORM
    else:
        raise InvalidQunatumDeviceError(f"Device {device} is not supported.")


def get_number_of_qubits(device: BaseDevice | object) -> int:
    """Get the number of qubits from the device.

    Args:
        device (BaseDevice | object): quantum device

    Returns:
        int: number of qubits
    """
    if isinstance(device, BaseDevice):
        return device.n_qubits

    if isinstance(device, PENNYLANE_DEVICES):
        return len(device.wires)  # type: ignore[attr-defined]
    else:
        raise InvalidQunatumDeviceError(f"Device {device} is not supported.")


def get_ibmq_available_devices(service: Any) -> list[RemoteDeviceStatus]:
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
    try:
        from braket.aws import AwsDevice
    except ImportError as exc:
        raise AmazonBraketSettingError(
            "Amazon Braket support requires optional dependencies. "
            'Install them with `pip install "qxmt[amazon-braket]"`.'
        ) from exc

    device_list = []
    devices = AwsDevice.get_devices()
    for device in devices:
        name = device.name
        n_qubits = device.properties.paradigm.qubitCount  # type: ignore
        status = STATUS_ONLINE if device.status == STATUS_ONLINE else STATUS_OFFLINE
        device_status = RemoteDeviceStatus(name=name, n_qubits=n_qubits, status=status)
        device_list.append(device_status)

    return device_list
