import os
from logging import Logger
from typing import Any, Optional

import numpy as np
from qiskit.providers.backend import BackendV2
from qiskit_ibm_runtime import IBMBackend

from qxmt.constants import IBMQ_API_KEY
from qxmt.exceptions import IBMQSettingError, InvalidPlatformError
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)
IBMQ_REAL_DEVICES = ["qiskit.remote"]


class BaseDevice:
    """General-purpose device class for experiment.
    This class is abstracted to oversee multiple platforms.
    Provide a common interface within the QXMT library by absorbing differences between platforms.

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
            backend_name (Optional[str]): backend name for the IBM Quantum real device
            n_qubits (int): number of qubits
            shots (Optional[int]): number of shots for the quantum circuit
            random_seed (Optional[int]): random seed for the quantum device
        """
        self.platform = platform
        self.device_name = device_name
        self.backend_name = backend_name
        self.n_qubits = n_qubits
        self.shots = shots
        self.random_seed = random_seed
        self.logger = logger
        self._set_device()

    def __call__(self) -> Any:
        return self.device

    def _get_ibmq_real_device(self, backend_name: Optional[str]) -> IBMBackend | BackendV2:
        """Get the IBM Quantum real device.
        This method accesses the IBM Quantum API,
        then please set the IBM Quantum account on environment variables "IBMQ_API_KEY".
        If the backend name is not provided, the least busy backend is selected.
        """
        from qiskit_ibm_runtime import QiskitRuntimeService

        ibm_api_key = os.getenv(IBMQ_API_KEY)
        if ibm_api_key is None:
            raise IBMQSettingError(
                f'IBM Quantum account is not set. Please set the "{IBMQ_API_KEY}" environment variable.'
            )

        QiskitRuntimeService.save_account(
            channel="ibm_quantum",
            token=ibm_api_key,
            overwrite=True,
            set_as_default=True,
        )

        service = QiskitRuntimeService()
        if backend_name is None:
            backend = service.least_busy(operational=True, simulator=False, min_num_qubits=self.n_qubits)
            self.logger.info(f'Backend is not provided. Select least busy backend: "{backend.name}"')
        else:
            backend = service.backend(backend_name)

        return backend

    def _set_ibmq_real_device_by_pennylane(self) -> None:
        """Set IBM Quantum real device by PennyLane."""
        import pennylane as qml

        backend = self._get_ibmq_real_device(self.backend_name)
        self.device = qml.device(
            name=self.device_name,
            backend=backend,
            wires=backend.num_qubits,
            shots=self.shots,
        )
        self.logger.info(
            "Set IBM Quantum real device: "
            f'(backend="{backend.name}", n_qubits={backend.num_qubits}, shots={self.shots})'
        )

    def _set_simulator_by_pennylane(self) -> None:
        """Set PennyLane simulator."""
        import pennylane as qml

        self.device = qml.device(
            name=self.device_name,
            wires=self.n_qubits,
            shots=self.shots,
            seed=np.random.default_rng(self.random_seed) if self.random_seed is not None else None,
        )

    def _set_device(self) -> None:
        """Set quantum device.

        Raises:
            InvalidPlatformError: platform is not implemented.
        """
        if self.platform == "pennylane":
            if self.device_name in IBMQ_REAL_DEVICES:
                self._set_ibmq_real_device_by_pennylane()
            else:
                self._set_simulator_by_pennylane()
        else:
            raise InvalidPlatformError(f'"{self.platform}" is not implemented.')

    def is_simulator(self) -> bool:
        """Check the device is a simulator or real machine.

        Returns:
            bool: True if the device is a simulator, False otherwise
        """
        return self.device_name not in IBMQ_REAL_DEVICES
