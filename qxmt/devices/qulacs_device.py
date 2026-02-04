from datetime import datetime
from typing import Any, Optional

from qxmt.devices.base import BaseDevice
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class QulacsDevice(BaseDevice):
    """Qulacs device implementation for quantum computation.
    This class provides a concrete implementation for Qulacs devices.
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
        """Initialize the Qulacs device."""
        super().__init__(platform, device_name, backend_name, n_qubits, shots, device_options, logger)
        self.real_device = None

    def get_device(self) -> "QulacsDevice":
        """Get the quantum device instance.
        Qulacs supports only a single type of simulator, and since it does not have a library-specific device class, the method returns itself.

        Returns:
            QulacsDevice: Qulacs device instance
        """
        return self

    def is_simulator(self) -> bool:
        """Check if the device is a simulator or real machine.

        Returns:
            bool: True if the device is a simulator, False otherwise
        """
        return True

    def is_remote(self) -> bool:
        """Check if the device is a remote device.
        Qulacs does not support remote devices.

        Returns:
            bool: True if the device is a remote device, False otherwise
        """
        return False

    def get_provider(self) -> str:
        """Get real machine provider name.
        Qulacs does not support remote devices.

        Returns:
            str: provider name (empty for non-remote devices)
        """
        return ""

    def get_backend_name(self) -> str:
        """Get real machine backend name.
        Qulacs does not support remote devices.

        Returns:
            str: backend name (empty for non-remote devices)
        """
        return ""

    def get_job_ids(
        self, created_after: Optional[datetime] = None, created_before: Optional[datetime] = None
    ) -> list[str]:
        """Get the job IDs.
        Qulacs does not support remote devices.

        Args:
            created_after (Optional[datetime]): created datetime of the jobs. If None, start time filter is not applied.
            created_before (Optional[datetime]): finished datetime of the jobs. If None, end time filter is not applied.

        Returns:
            list[str]: job IDs
        """
        return []
