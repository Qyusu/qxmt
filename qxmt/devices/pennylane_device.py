import os
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pennylane as qml

from qxmt.devices.base import BaseDevice, LOGGER
from qxmt.devices.amazon import (
    AMAZON_BRACKET_DEVICES,
    AMAZON_BRACKET_LOCAL_BACKENDS,
    AMAZON_BRACKET_REMOTE_DEVICES,
    AMAZON_BRAKET_DEVICES,
    AMAZON_BRAKET_SIMULATOR_BACKENDS,
    AMAZON_PROVIDER_NAME,
)
from qxmt.devices.ibmq import IBMQ_PROVIDER_NAME, IBMQ_REAL_DEVICES
from qxmt.exceptions import (
    AmazonBraketSettingError,
    IBMQSettingError,
    InvalidPlatformError,
)


class PennyLaneDevice(BaseDevice):
    """PennyLane device implementation for quantum computation.
    This class provides a concrete implementation of the BaseDevice for PennyLane.
    """

    def __init__(
        self,
        platform: str,
        device_name: str,
        backend_name: Optional[str],
        n_qubits: int,
        shots: Optional[int],
        random_seed: Optional[int] = None,
        logger: Any = LOGGER,
    ) -> None:
        """Initialize the PennyLane device.

        Args:
            platform (str): platform name (ex: pennylane, qulacs, etc.)
            device_name (str): device name provided by the platform (ex: default.qubit, default.tensor, etc.)
            backend_name (Optional[str]): backend name for the real device
            n_qubits (int): number of qubits
            shots (Optional[int]): number of shots for the quantum circuit
            random_seed (Optional[int]): random seed for the quantum device
            logger (Any): logger instance
        """
        super().__init__(platform, device_name, backend_name, n_qubits, shots, random_seed, logger)
        self.real_device = None

    def get_device(self) -> Any:
        """Get the quantum device instance.

        Returns:
            Any: quantum device instance
        """
        return qml.device(
            name=self.device_name,
            wires=self.n_qubits,
            shots=self.shots,
            seed=np.random.default_rng(self.random_seed) if self.random_seed is not None else None,
        )

    def is_simulator(self) -> bool:
        """Check if the device is a simulator or real machine.

        Returns:
            bool: True if the device is a simulator, False otherwise
        """
        return True

    def is_remote(self) -> bool:
        """Check if the device is a remote device.

        Returns:
            bool: True if the device is a remote device, False otherwise
        """
        return False
