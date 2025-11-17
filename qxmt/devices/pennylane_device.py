from datetime import datetime
from typing import Any, Optional

import pennylane as qml

from qxmt.devices.base import BaseDevice
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class PennyLaneDevice(BaseDevice):
    """PennyLane device implementation for quantum computation.
    This class provides a concrete implementation for PennyLane devices.
    """

    def __init__(
        self,
        platform: str,
        device_name: str,
        backend_name: Optional[str],
        n_qubits: int,
        shots: Optional[int],
        device_options: Optional[dict[str, Any]] = None,
        logger: Any = LOGGER,
    ) -> None:
        """Initialize the PennyLane device.

        Args:
            platform (str): platform name (ex: pennylane, qulacs, etc.)
            device_name (str): device name provided by the platform (ex: default.qubit, default.tensor, etc.)
            backend_name (Optional[str]): backend name for the real device
            n_qubits (int): number of qubits
            shots (Optional[int]): number of shots for the quantum circuit
            device_options (Optional[dict[str, Any]]): additional keyword arguments for qml.device
            logger (Any): logger instance
        """
        super().__init__(platform, device_name, backend_name, n_qubits, shots, device_options, logger)
        self.real_device = None
        self.device_options: dict[str, Any] = dict(device_options) if device_options is not None else {}
        self._validate_device_options()

    def get_device(self) -> Any:
        """Get the quantum device instance.

        Returns:
            Any: quantum device instance
        """
        device_kwargs = self._build_device_kwargs()
        return qml.device(name=self.device_name, **device_kwargs)

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

    def get_provider(self) -> str:
        """Get real machine provider name.

        Returns:
            str: provider name (empty for non-remote devices)
        """
        return ""

    def get_backend_name(self) -> str:
        """Get real machine backend name.

        Returns:
            str: backend name (empty for non-remote devices)
        """
        return ""

    def get_job_ids(
        self, created_after: Optional[datetime] = None, created_before: Optional[datetime] = None
    ) -> list[str]:
        """Get the job IDs.
        Local machine does not have job IDs.

        Args:
            created_after (Optional[datetime]): created datetime of the jobs. If None, start time filter is not applied.
            created_before (Optional[datetime]): finished datetime of the jobs. If None, end time filter is not applied.

        Returns:
            list[str]: job IDs (empty for non-remote devices)
        """
        return []

    def _validate_device_options(self) -> None:
        invalid_keys = {"name", "wires"}
        duplicated_keys = invalid_keys.intersection(self.device_options)
        if duplicated_keys:
            joined = ", ".join(sorted(duplicated_keys))
            raise ValueError(f'"device_options" cannot override the following keys: {joined}')

    def _build_device_kwargs(self) -> dict[str, Any]:
        device_kwargs: dict[str, Any] = {
            "wires": self.n_qubits,
            "shots": self.shots,
        }
        extra_options = dict(self.device_options)
        device_kwargs.update(extra_options)
        return device_kwargs
