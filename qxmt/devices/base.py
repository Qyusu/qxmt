import os
from datetime import datetime
from enum import Enum
from logging import Logger
from typing import Any, Optional

import numpy as np
import pennylane as qml
from braket.devices import Devices
from qiskit.providers.backend import BackendV2
from qiskit_ibm_runtime import IBMBackend, QiskitRuntimeService

from qxmt.constants import IBMQ_API_KEY
from qxmt.exceptions import (
    AmazonBraketSettingError,
    IBMQSettingError,
    InvalidPlatformError,
)
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)

IBMQ_PROVIDER_NAME = "IBM_Quantum"
AMAZON_PROVIDER_NAME = "Amazon_Braket"
IBMQ_REAL_DEVICES = ["qiskit.remote"]
AMAZON_BRACKETS_DEVICES = ["braket.local.qubit"]
AMAZON_BRACKETS_REMOTE_DEVICES = ["braket.aws.qubit"]


class AmazonBackendType(Enum):
    sv1 = Devices.Amazon.SV1
    dm1 = Devices.Amazon.DM1
    tn1 = Devices.Amazon.TN1


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
        self.real_device = None

    def _get_ibmq_real_device(self, backend_name: Optional[str]) -> IBMBackend | BackendV2:
        """Get the IBM Quantum real device.
        This method accesses the IBM Quantum API,
        then please set the IBM Quantum account on environment variables "IBMQ_API_KEY".
        If the backend name is not provided, the least busy backend is selected.
        """
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

    def _get_amazon_remote_device_by_pennylane(self) -> Any:
        if self.backend_name is None:
            raise IBMQSettingError("Amazon Braket device needs the backend name.")

        try:
            device_arn = AmazonBackendType[self.backend_name.lower()].value
        except KeyError:
            raise AmazonBraketSettingError(f'"{self.backend_name}" is not supported Amazon Braket device.')

        return qml.device(
            name=self.device_name,
            device_arn=device_arn.value,
            wires=self.n_qubits,
            shots=self.shots,
        )

    def _get_simulator_by_pennylane(self) -> Any:
        """Set PennyLane simulator."""
        if self.device_name in AMAZON_BRACKETS_DEVICES:
            # Amazon Braket local simulator
            # Amazon Braket device not support the random seed
            return qml.device(
                name=self.device_name,
                wires=self.n_qubits,
                backend="braket_sv",
                shots=self.shots,
            )
        elif self.device_name in AMAZON_BRACKETS_REMOTE_DEVICES:
            # Amazon Braket remote simulator
            # Amazon Braket device not support the random seed
            return self._get_amazon_remote_device_by_pennylane()
        else:
            # PennyLane original simulator
            return qml.device(
                name=self.device_name,
                wires=self.n_qubits,
                shots=self.shots,
                seed=np.random.default_rng(self.random_seed) if self.random_seed is not None else None,
            )

    def get_device(self) -> Any:
        if self.platform == "pennylane":
            if (self.device_name in IBMQ_REAL_DEVICES) and (self.real_device is None):
                self._set_ibmq_real_device_by_pennylane()
                return self.real_device
            elif (self.device_name in IBMQ_REAL_DEVICES) and (self.real_device is not None):
                return self.real_device
            else:
                return self._get_simulator_by_pennylane()
        else:
            raise InvalidPlatformError(f'"{self.platform}" is not implemented.')

    def is_simulator(self) -> bool:
        """Check the device is a simulator or real machine.

        Returns:
            bool: True if the device is a simulator, False otherwise
        """
        return self.device_name not in IBMQ_REAL_DEVICES

    def is_ibmq_device(self) -> bool:
        """Check the device is an IBM Quantum device.

        Returns:
            bool: True if the device is an IBM Quantum device, False otherwise
        """
        return self.device_name in IBMQ_REAL_DEVICES

    def is_amazon_device(self) -> bool:
        """Check the device is an Amazon Braket device.

        Returns:
            bool: True if the device is an Amazon Braket device, False otherwise
        """
        return self.device_name in AMAZON_BRACKETS_DEVICES + AMAZON_BRACKETS_REMOTE_DEVICES

    def get_provider(self) -> str:
        """Get real machine provider name.

        Returns:
            str: provider name
        """
        if self.is_ibmq_device():
            return IBMQ_PROVIDER_NAME
        elif self.is_amazon_device():
            return AMAZON_PROVIDER_NAME
        else:
            return ""

    def get_service(self) -> QiskitRuntimeService:
        """Get the IBM Quantum service.

        Returns:
            QiskitRuntimeService: IBM Quantum service
        """
        if self.is_simulator():
            raise IBMQSettingError(f'The device ("{self.device_name}") is a simulator.')

        if self.real_device is None:
            raise IBMQSettingError(f'The device ("{self.device_name}") is not set.')

        return self.real_device.service

    def get_backend(self) -> IBMBackend | BackendV2:
        """Get the IBM Quantum real device backend.

        Returns:
            IBMBackend | BackendV2: IBM Quantum real device backend
        """
        if self.is_simulator():
            raise IBMQSettingError(f'The device ("{self.device_name}") is a simulator.')

        if self.real_device is None:
            raise IBMQSettingError(f'The device ("{self.device_name}") is not set.')

        return self.real_device.backend

    def get_backend_name(self) -> str:
        """Get the IBM Quantum real device backend name.

        Returns:
            str: IBM Quantum real device backend name
        """
        backend = self.get_backend()
        return backend.name

    def get_ibmq_job_ids(
        self, created_after: Optional[datetime] = None, created_before: Optional[datetime] = None
    ) -> list[str]:
        """Get the IBM Quantum job IDs.

        Returns:
            list[str]: IBM Quantum job IDs
        """
        if self.is_simulator():
            raise IBMQSettingError(f'The device ("{self.device_name}") is a simulator.')

        service = self.get_service()
        backend = self.get_backend()
        jobs = service.jobs(backend_name=backend.name, created_after=created_after, created_before=created_before)

        return [job.job_id() for job in jobs]
