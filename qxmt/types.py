from typing import Optional, TypeVar

import numpy as np
import pennylane as qml

# [TODO]: constract from qxmt.constants
QuantumDeviceType = TypeVar("QuantumDeviceType", qml.devices.Device, qml.devices.LegacyDevice, qml.devices.QubitDevice)


RAW_DATA_TYPE = np.ndarray
RAW_LABEL_TYPE = np.ndarray
RAW_DATASET_TYPE = tuple[RAW_DATA_TYPE, RAW_LABEL_TYPE]
PROCESSCED_DATASET_TYPE = tuple[
    np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray
]
