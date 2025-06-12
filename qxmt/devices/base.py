from abc import ABC, abstractmethod
from datetime import datetime
from logging import Logger
from typing import Any, Literal, Optional

from qxmt.devices.amazon import (
    AMAZON_BRAKET_DEVICES,
    AMAZON_BRAKET_LOCAL_DEVICES,
    AMAZON_BRAKET_REMOTE_DEVICES,
)
from qxmt.devices.ibmq import IBMQ_REAL_DEVICES
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class BaseDevice(ABC):
    """Abstract base class for quantum devices."""

    def __init__(
        self,
        platform: str,
        device_name: str,
        backend_name: Optional[str],
        n_qubits: int,
        shots: Optional[int],
        random_seed: Optional[int] = None,
        logger: Logger = LOGGER,
    ) -> None:
        """Initialize the quantum device.

        Args:
            platform (str): platform name (ex: pennylane, qulacs, etc.)
            device_name (str): device name provided by the platform (ex: default.qubit, default.tensor, etc.)
            backend_name (Optional[str]): backend name for the real device
            n_qubits (int): number of qubits
            shots (Optional[int]): number of shots for the quantum circuit
            random_seed (Optional[int]): random seed for the quantum device
            logger (Logger): logger instance
        """
        self.platform = platform
        self.device_name = device_name
        self.backend_name = backend_name
        self.n_qubits = n_qubits
        self.shots = shots
        self.random_seed = random_seed
        self.logger = logger

    @abstractmethod
    def get_device(self) -> Any:
        """Get the quantum device instance.

        Returns:
            Any: quantum device instance
        """
        pass

    @abstractmethod
    def is_simulator(self) -> bool:
        """Check if the device is a simulator or real machine.

        Returns:
            bool: True if the device is a simulator, False otherwise
        """
        pass

    @abstractmethod
    def is_remote(self) -> bool:
        """Check if the device is a remote device.

        Returns:
            bool: True if the device is a remote device, False otherwise
        """
        pass

    @abstractmethod
    def get_provider(self) -> str:
        """Get real machine provider name.

        Returns:
            str: provider name
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Get real machine backend name.

        Returns:
            str: backend name
        """
        pass

    @abstractmethod
    def get_job_ids(
        self, created_after: Optional[datetime] = None, created_before: Optional[datetime] = None
    ) -> list[str]:
        """Get the job IDs.

        Args:
            created_after (Optional[datetime]): created datetime of the jobs. If None, start time filter is not applied.
            created_before (Optional[datetime]): finished datetime of the jobs. If None, end time filter is not applied.

        Returns:
            list[str]: job IDs
        """
        pass

    def is_ibmq_device(self) -> bool:
        """Check if the device is an IBM Quantum device.

        Returns:
            bool: True if the device is an IBM Quantum device, False otherwise
        """
        return self.device_name in IBMQ_REAL_DEVICES

    def is_amazon_device(self, device_type: Literal["local", "remote", "all"] = "all") -> bool:
        """Check if the device is an Amazon Braket device.

        Args:
            device_type (Literal["local", "remote", "all"]): type of Amazon Braket device

        Returns:
            bool: True if the device is an Amazon Braket device, False otherwise
        """
        if device_type == "local":
            return self.device_name in AMAZON_BRAKET_LOCAL_DEVICES
        elif device_type == "remote":
            return self.device_name in AMAZON_BRAKET_REMOTE_DEVICES
        elif device_type == "all":
            return self.device_name in AMAZON_BRAKET_DEVICES
