from datetime import datetime
from typing import Any, Optional

from qxmt.devices.base import BaseDevice
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class QiskitDevice(BaseDevice):
    """Qiskit Aer simulator device implementation."""

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
        """Initialize the Qiskit device.

        Args:
            platform (str): platform name.
            device_name (str): AerSimulator method name.
            backend_name (Optional[str]): backend name for the real device.
            n_qubits (int): number of qubits.
            shots (Optional[int]): number of shots for circuit execution.
            device_options (Optional[dict[str, Any]]): additional keyword arguments for AerSimulator.
            logger (Any): logger instance.
        """
        super().__init__(platform, device_name, backend_name, n_qubits, shots, device_options, logger)
        self.real_device = None
        self.default_kwargs = {
            "method": self.device_name,
        }
        self._validate_device_options(invalid_keys=set(self.default_kwargs.keys()))

    def get_device(self) -> Any:
        """Get the Qiskit Aer simulator instance.

        Returns:
            Any: Qiskit AerSimulator instance.
        """
        from qiskit_aer import AerSimulator

        device_kwargs = self._build_device_kwargs(default_kwargs=self.default_kwargs)
        return AerSimulator(**device_kwargs)

    def is_simulator(self) -> bool:
        """Check if the device is a simulator or real machine.

        Returns:
            bool: True for Qiskit Aer simulator.
        """
        return True

    def is_remote(self) -> bool:
        """Check if the device is a remote device.

        Returns:
            bool: False for Qiskit Aer simulator.
        """
        return False

    def get_provider(self) -> str:
        """Get real machine provider name.

        Returns:
            str: provider name (empty for non-remote devices).
        """
        return ""

    def get_backend_name(self) -> str:
        """Get real machine backend name.

        Returns:
            str: backend name (empty for non-remote devices).
        """
        return ""

    def get_job_ids(
        self, created_after: Optional[datetime] = None, created_before: Optional[datetime] = None
    ) -> list[str]:
        """Get the job IDs.

        Args:
            created_after (Optional[datetime]): created datetime of the jobs.
            created_before (Optional[datetime]): finished datetime of the jobs.

        Returns:
            list[str]: job IDs (empty for local simulator devices).
        """
        return []
