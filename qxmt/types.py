from typing import TypeVar

import pennylane as qml

# [TODO]: constract from qxmt.constants
QuantumDeviceType = TypeVar("QuantumDeviceType", qml.devices.Device, qml.Device, qml.QubitDevice)
