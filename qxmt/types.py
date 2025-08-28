from typing import Optional, TypeVar

import numpy as np

from qxmt.constants import PENNYLANE_DEVICES

if PENNYLANE_DEVICES:
    QuantumDeviceType = TypeVar("QuantumDeviceType", *PENNYLANE_DEVICES)  # type: ignore[misc]
else:
    QuantumDeviceType = TypeVar("QuantumDeviceType")


RAW_DATA_TYPE = np.ndarray
RAW_LABEL_TYPE = np.ndarray
RAW_DATASET_TYPE = tuple[RAW_DATA_TYPE, RAW_LABEL_TYPE]
PROCESSCED_DATASET_TYPE = tuple[
    np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray
]
