from typing import TYPE_CHECKING, Any

from qxmt.devices.base import BaseDevice
from qxmt.devices.builder import DeviceBuilder
from qxmt.devices.device_info import get_number_of_qubits, get_platform_from_device
from qxmt.devices.pennylane_device import PennyLaneDevice

if TYPE_CHECKING:
    from qxmt.devices.amazon_device import AmazonBraketDevice
    from qxmt.devices.ibmq_device import IBMQDevice
    from qxmt.devices.qulacs_device import QulacsDevice

__all__ = [
    "AmazonBraketDevice",
    "BaseDevice",
    "DeviceBuilder",
    "get_number_of_qubits",
    "get_platform_from_device",
    "IBMQDevice",
    "PennyLaneDevice",
    "QulacsDevice",
]

_OPTIONAL_DEVICE_IMPORTS = {
    "AmazonBraketDevice": (
        "qxmt.devices.amazon_device",
        "amazon-braket",
        'pip install "qxmt[amazon-braket]"',
    ),
    "IBMQDevice": (
        "qxmt.devices.ibmq_device",
        "pennylane-qiskit",
        'pip install "qxmt[pennylane-qiskit]"',
    ),
    "QulacsDevice": (
        "qxmt.devices.qulacs_device",
        "qulacs",
        'pip install "qxmt[qulacs]"',
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _OPTIONAL_DEVICE_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, extra_name, install_command = _OPTIONAL_DEVICE_IMPORTS[name]
    try:
        module = __import__(module_name, fromlist=[name])
    except ImportError as exc:
        raise ImportError(
            f"{name} requires the optional '{extra_name}' dependencies. Install them with `{install_command}`."
        ) from exc

    value = getattr(module, name)
    globals()[name] = value
    return value
