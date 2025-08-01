from datetime import datetime
from typing import Any, Optional

import numpy as np
import pennylane as qml

from qxmt.devices.base import BaseDevice
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)

PENNYLANE_GPU_DEVICES = ["lightning.gpu", "lightning.tensor"]


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
        if self.device_name in PENNYLANE_GPU_DEVICES:
            return self._get_gpu_device()
        else:
            return self._get_cpu_device()

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

    def _get_cpu_device(self) -> Any:
        return qml.device(
            name=self.device_name,
            wires=self.n_qubits,
            shots=self.shots,
            seed=np.random.default_rng(self.random_seed) if self.random_seed is not None else None,
        )

    def _get_gpu_device(self) -> Any:
        return qml.device(
            name=self.device_name,
            wires=self.n_qubits,
            shots=self.shots,
        )
