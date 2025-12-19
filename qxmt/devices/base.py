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
        device_options: Optional[dict[str, Any]] = None,
        logger: Logger = LOGGER,
    ) -> None:
        """Initialize the quantum device.

        Args:
            platform (str): platform name (ex: pennylane, qulacs, etc.)
            device_name (str): device name provided by the platform (ex: default.qubit, default.tensor, etc.)
            backend_name (Optional[str]): backend name for the real device
            n_qubits (int): number of qubits
            shots (Optional[int]): number of shots for the quantum circuit
            device_options (Optional[dict[str, Any]]): additional keyword arguments for the device
            logger (Logger): logger instance
        """
        self.platform = platform
        self.device_name = device_name
        self.backend_name = backend_name
        self.n_qubits = n_qubits
        self.shots = shots
        self.device_options: dict[str, Any] = dict(device_options) if device_options is not None else {}
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

    def _validate_device_options(self, invalid_keys: set[str] = set()) -> None:
        """Validate device options.

        Args:
            invalid_keys (set[str]): set of keys that should not be in device_options
        """
        duplicated_keys = invalid_keys.intersection(self.device_options)
        if duplicated_keys:
            joined = ", ".join(sorted(duplicated_keys))
            raise ValueError(f'"device_options" cannot override the following keys: {joined}')

    def _build_device_kwargs(self, default_kwargs: dict[str, Any] = {}) -> dict[str, Any]:
        """Build keyward argments for the device.

        Args:
            default_kwargs (dict[str, Any], optional): default keywards. Defaults to {}.

        Returns:
            dict[str, Any]: constructed keyward argments for the device
        """
        device_kwargs: dict[str, Any] = default_kwargs.copy()
        extra_options = dict(self.device_options)
        device_kwargs.update(extra_options)
        return device_kwargs
