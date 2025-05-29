import os
from datetime import datetime
from typing import Any, Optional

import pennylane as qml
from qiskit.providers.backend import BackendV2
from qiskit_ibm_runtime import IBMBackend, QiskitRuntimeService

from qxmt.constants import IBMQ_API_KEY
from qxmt.devices.base import BaseDevice
from qxmt.devices.ibmq import IBMQ_PROVIDER_NAME
from qxmt.exceptions import IBMQSettingError
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class IBMQDevice(BaseDevice):
    """IBMQ device implementation for quantum computation.
    This class provides a concrete implementation of the BaseDevice for IBM Quantum.
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
        """Initialize the IBMQ device.

        Args:
            platform (str): platform name (ex: pennylane, qulacs, etc.)
            device_name (str): device name provided by the platform (ex: default.qubit, default.tensor, etc.)
            backend_name (Optional[str]): backend name for the IBM Quantum real device
            n_qubits (int): number of qubits
            shots (Optional[int]): number of shots for the quantum circuit
            random_seed (Optional[int]): random seed for the quantum device
            logger (Any): logger instance
        """
        super().__init__(platform, device_name, backend_name, n_qubits, shots, random_seed, logger)
        self.real_device = None
        self.ibm_api_key = None
        self._set_ibmq_settings()

    def _set_ibmq_settings(self) -> None:
        """Set the IBM Quantum account settings.
        This method check the IBM Quantum account settings on the environment variables.

        Raises:
            IBMQSettingError: IBM Quantum API key is not set to the environment variables
        """
        self.ibm_api_key = os.getenv(IBMQ_API_KEY)
        if self.ibm_api_key is None:
            raise IBMQSettingError(f"Environmet variable for IBMQ not set: {IBMQ_API_KEY}")

    def _check_ibmq_availability(self, backend: IBMBackend | BackendV2) -> None:
        """Check the IBM Quantum real device availability.
        This method checks the device status and the number of qubits.

        Args:
            backend (IBMBackend | BackendV2): IBM Quantum real device backend
        """
        if isinstance(backend, IBMBackend):
            is_online = backend.status().operational
        elif isinstance(backend, BackendV2):
            is_online = True

        max_qubits = backend.num_qubits
        is_enough_qubits = self.n_qubits <= max_qubits

        if not (is_online and is_enough_qubits):
            raise IBMQSettingError(
                f'The device ("{self.device_name}") is not available. '
                f"Please check the device status and the number of qubits. "
                f"(is_online={is_online}, is_enough_qubits={is_enough_qubits} (max_qubits={max_qubits}))"
            )

    def _get_ibmq_real_device(self, backend_name: Optional[str]) -> IBMBackend | BackendV2:
        """Get the IBM Quantum real device.
        This method accesses the IBM Quantum API,
        then please set the IBM Quantum account on environment variables "IBMQ_API_KEY".
        If the backend name is not provided, the least busy backend is selected.

        Args:
            backend_name (Optional[str]): backend name for the IBM Quantum real device

        Returns:
            IBMBackend | BackendV2: IBM Quantum real device backend
        """
        QiskitRuntimeService.save_account(
            channel="ibm_quantum",
            token=self.ibm_api_key,
            overwrite=True,
            set_as_default=True,
        )

        service = QiskitRuntimeService()
        if backend_name is None:
            backend = service.least_busy(operational=True, simulator=False, min_num_qubits=self.n_qubits)
            self.logger.info(f'Backend is not provided. Select least busy backend: "{backend.name}"')
        else:
            backend = service.backend(backend_name)

        self._check_ibmq_availability(backend)

        return backend

    def _set_ibmq_real_device_by_pennylane(self) -> None:
        """Set IBM Quantum real device by PennyLane."""
        if self.shots is None:
            raise IBMQSettingError("Real quantum machine must set the shots.")

        backend = self._get_ibmq_real_device(self.backend_name)
        self.real_device = qml.device(
            name=self.device_name,
            backend=backend,
            wires=backend.num_qubits,
            shots=self.shots,
        )
        self.logger.info(
            "Set IBM Quantum real device: "
            f'(backend="{backend.name}", n_qubits={backend.num_qubits}, shots={self.shots})'
        )

    def get_device(self) -> Any:
        """Get the quantum device instance.

        Returns:
            Any: quantum device instance
        """
        if self.real_device is None:
            self._set_ibmq_real_device_by_pennylane()
            return self.real_device
        else:
            return self.real_device

    def is_simulator(self) -> bool:
        """Check if the device is a simulator or real machine.

        Returns:
            bool: True if the device is a simulator, False otherwise
        """
        return False

    def is_remote(self) -> bool:
        """Check if the device is a remote device.

        Returns:
            bool: True if the device is a remote device, False otherwise
        """
        return True

    def get_provider(self) -> str:
        """Get real machine provider name.

        Returns:
            str: provider name
        """
        return IBMQ_PROVIDER_NAME

    def get_service(self) -> QiskitRuntimeService:
        """Get the IBM Quantum service.

        Returns:
            QiskitRuntimeService: IBM Quantum service

        Raises:
            IBMQSettingError: The real device is not set
        """
        if self.real_device is None:
            raise IBMQSettingError(f'The real device ("{self.device_name}") is not set.')

        return self.real_device.service

    def get_backend(self) -> IBMBackend | BackendV2:
        """Get the IBM Quantum real device backend.

        Returns:
            IBMBackend | BackendV2: IBM Quantum real device backend

        Raises:
            IBMQSettingError: The real device is not set
        """
        if self.real_device is None:
            raise IBMQSettingError(f'The real device ("{self.device_name}") is not set.')

        return self.real_device.backend

    def get_backend_name(self) -> str:
        """Get the real or remote backend name.

        Returns:
            str: backend name
        """
        backend = self.get_backend()
        return backend.name

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
        service = self.get_service()
        backend = self.get_backend()
        jobs = service.jobs(backend_name=backend.name, created_after=created_after, created_before=created_before)

        return [job.job_id() for job in jobs]
