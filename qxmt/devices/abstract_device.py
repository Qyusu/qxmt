from abc import ABC, abstractmethod
from datetime import datetime
from logging import Logger
from typing import Any, Optional

from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class AbstractDevice(ABC):
    """Abstract base class for quantum devices across different platforms.
    This class defines the common interface for all quantum device implementations.
    Concrete implementations should inherit from this class and implement the abstract methods.
    """

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
