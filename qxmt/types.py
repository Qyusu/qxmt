from typing import TypeVar, Union

import pennylane as qml

# [TODO]: constract from qxmt.constants
QuantumDeviceType = TypeVar("QuantumDeviceType", qml.devices.Device, qml.Device, qml.QubitDevice)
