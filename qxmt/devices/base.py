from datetime import datetime
from logging import Logger
from typing import Any, Literal, Optional

from qxmt.devices.amazon import (
    AMAZON_BRACKET_DEVICES,
    AMAZON_BRACKET_LOCAL_BACKENDS,
    AMAZON_BRACKET_REMOTE_DEVICES,
    AMAZON_BRAKET_DEVICES,
    AMAZON_BRAKET_SIMULATOR_BACKENDS,
    AMAZON_PROVIDER_NAME,
)
from qxmt.devices.ibmq import IBMQ_PROVIDER_NAME, IBMQ_REAL_DEVICES
from qxmt.exceptions import InvalidPlatformError
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class BaseDevice:
    """General-purpose device class for experiment.
    This class is abstracted to oversee multiple platforms.
    Provide a common interface within the QXMT library by absorbing differences between platforms.

    This class maintains backward compatibility with the previous implementation
    while delegating to the appropriate concrete implementation classes.

    Examples:
        >>> from qxmt.devices.base import BaseDevice
        >>> device = BaseDevice(
        ...     platform="pennylane",
        ...     device_name="default.qubit",
        ...     backend_name=None,
        ...     n_qubits=2,
        ...     shots=100,
        ...     random_seed=42,
        ...     )
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

        if platform == "pennylane":
            if device_name in IBMQ_REAL_DEVICES:
                from qxmt.devices.ibmq_device import IBMQDevice

                self._impl = IBMQDevice(platform, device_name, backend_name, n_qubits, shots, random_seed, logger)
            elif device_name in AMAZON_BRAKET_DEVICES:
                from qxmt.devices.amazon_device import AmazonBraketDevice

                self._impl = AmazonBraketDevice(
                    platform, device_name, backend_name, n_qubits, shots, random_seed, logger
                )
            else:
                from qxmt.devices.pennylane_device import PennyLaneDevice

                self._impl = PennyLaneDevice(platform, device_name, backend_name, n_qubits, shots, random_seed, logger)
        else:
            raise InvalidPlatformError(f'"{platform}" is not implemented.')

    def get_device(self) -> Any:
        """Get the quantum device instance.

        Returns:
            Any: quantum device instance
        """
        return self._impl.get_device()

    def is_simulator(self) -> bool:
        """Check if the device is a simulator or real machine.

        Returns:
            bool: True if the device is a simulator, False otherwise
        """
        return self._impl.is_simulator()

    def is_remote(self) -> bool:
        """Check if the device is a remote device.

        Returns:
            bool: True if the device is a remote device, False otherwise
        """
        return self._impl.is_remote()

    def get_provider(self) -> str:
        """Get real machine provider name.

        Returns:
            str: provider name
        """
        return self._impl.get_provider()

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
        return self._impl.get_job_ids(created_after, created_before)

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
            return self.device_name in AMAZON_BRACKET_LOCAL_BACKENDS
        elif device_type == "remote":
            return self.device_name in AMAZON_BRACKET_REMOTE_DEVICES
        elif device_type == "all":
            return self.device_name in AMAZON_BRAKET_DEVICES

    def _get_amazon_local_simulator_by_pennylane(self) -> Any:
        """Get Amazon Braket local simulator by PennyLane.

        Returns:
            Any: quantum device instance for Amazon Braket local simulator
        """
        if not self.is_amazon_device(device_type="local"):
            raise ValueError("This method is only available for Amazon Braket local simulator devices.")

        return (
            self._impl._get_amazon_local_simulator_by_pennylane()
            if hasattr(self._impl, "_get_amazon_local_simulator_by_pennylane")
            else None
        )

    def _get_amazon_remote_device_by_pennylane(self) -> Any:
        """Get Amazon Braket remote device by PennyLane.

        Returns:
            Any: quantum device instance for Amazon Braket remote device
        """
        if not self.is_amazon_device(device_type="remote"):
            raise ValueError("This method is only available for Amazon Braket remote devices.")

        return (
            self._impl._get_amazon_remote_device_by_pennylane()
            if hasattr(self._impl, "_get_amazon_remote_device_by_pennylane")
            else None
        )

    def get_service(self) -> Any:
        """Get the IBM Quantum service.

        Returns:
            Any: IBM Quantum service

        Raises:
            ValueError: This method is only available for IBM Quantum devices
        """
        if not self.is_ibmq_device():
            raise ValueError("This method is only available for IBM Quantum devices.")

        return self._impl.get_service() if hasattr(self._impl, "get_service") else None

    def get_backend(self) -> Any:
        """Get the IBM Quantum real device backend.

        Returns:
            Any: IBM Quantum real device backend

        Raises:
            ValueError: This method is only available for IBM Quantum devices
        """
        if not self.is_ibmq_device():
            raise ValueError("This method is only available for IBM Quantum devices.")

        return self._impl.get_backend() if hasattr(self._impl, "get_backend") else None

    def get_backend_name(self) -> str:
        """Get the real or remote backend name.

        Returns:
            str: backend name
        """
        if hasattr(self._impl, "get_backend_name"):
            return self._impl.get_backend_name()
        elif self.backend_name is None:
            raise ValueError("The backend name is not set.")
        return self.backend_name
