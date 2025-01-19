import os
from datetime import datetime, timezone
from logging import Logger
from typing import Any, Literal, Optional

import boto3
import numpy as np
import pennylane as qml
from braket.aws import AwsDevice
from qiskit.providers.backend import BackendV2
from qiskit_ibm_runtime import IBMBackend, QiskitRuntimeService

from qxmt.constants import (
    AWS_ACCESS_KEY_ID,
    AWS_DEFAULT_REGION,
    AWS_SECRET_ACCESS_KEY,
    IBMQ_API_KEY,
)
from qxmt.devices.amazon import (
    AMAZON_BRACKET_DEVICES,
    AMAZON_BRACKET_LOCAL_BACKENDS,
    AMAZON_BRACKET_REMOTE_DEVICES,
    AMAZON_BRAKET_DEVICES,
    AMAZON_BRAKET_SIMULATOR_BACKENDS,
    AMAZON_PROVIDER_NAME,
    AmazonBackendType,
)
from qxmt.devices.ibmq import IBMQ_PROVIDER_NAME, IBMQ_REAL_DEVICES
from qxmt.exceptions import (
    AmazonBraketSettingError,
    IBMQSettingError,
    InvalidPlatformError,
)
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


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
        self.ibm_api_key = None
        self.aws_access_key_id = None
        self.aws_secret_access_key = None
        self.aws_default_region = None

        if self.is_ibmq_device():
            self._set_ibmq_settings()

        if self.is_amazon_device(device_type="remote"):
            self._set_amazon_braket_settings()

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
            # BackendV2 does not have the status method, then always return True
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

    def _set_amazon_braket_settings(self) -> None:
        """Set the Amazon Braket account settings.
        This method check the Amazon Braket account settings on the environment variables.

        Raises:
            AmazonBraketSettingError: Amazon Braket account settings are not set to the environment variables
        """
        self.aws_access_key_id = os.getenv(AWS_ACCESS_KEY_ID)
        self.aws_secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY)
        self.aws_default_region = os.getenv(AWS_DEFAULT_REGION)

        missing = [
            name
            for name, value in [
                (AWS_ACCESS_KEY_ID, self.aws_access_key_id),
                (AWS_SECRET_ACCESS_KEY, self.aws_secret_access_key),
                (AWS_DEFAULT_REGION, self.aws_default_region),
            ]
            if value is None
        ]

        if missing:
            raise AmazonBraketSettingError(f"Environment variables for Amazon Braket not set: {', '.join(missing)}")

    def _check_amazon_braket_availability(self, device: AwsDevice) -> None:
        """Check the Amazon Braket device availability.
        This method checks the device status and the number of qubits.

        Args:
            device (AwsDevice): Amazon Braket device instance
        """
        is_online = device.status == "ONLINE"
        max_qubits = int(device.properties.paradigm.qubitCount)  # type: ignore
        is_enough_qubits = bool(self.n_qubits <= max_qubits)
        if not (is_online and is_enough_qubits):
            raise AmazonBraketSettingError(
                f'The device ("{self.device_name}") is not available. '
                f"Please check the device status and the number of qubits. "
                f"(is_online={is_online}, is_enough_qubits={is_enough_qubits} (max_qubits={max_qubits}))"
            )

    def _get_amazon_local_simulator_by_pennylane(self) -> Any:
        """Get Amazon Braket local simulator by PennyLane.

        Returns:
            Any: quantum device instance for Amazon Braket local simulator
        """
        if self.backend_name is None:
            self.logger.info("Backend name is not provided. Select the state vector backend.")
            self.backend_name = "braket_sv"

        if self.backend_name not in AMAZON_BRACKET_LOCAL_BACKENDS:
            raise AmazonBraketSettingError(f'"{self.backend_name}" is not supported Amazon Braket local simulator.')

        # Amazon Braket device not support the random seed
        return qml.device(
            name=self.device_name,
            wires=self.n_qubits,
            backend=self.backend_name,
            shots=self.shots,
        )

    def _get_amazon_remote_device_by_pennylane(self) -> Any:
        """Get Amazon Braket remote device by PennyLane.

        Returns:
            Any: quantum device instance for Amazon Braket remote device
        """
        if self.backend_name is None:
            raise AmazonBraketSettingError("Amazon Braket device needs the backend name.")

        try:
            device_arn = AmazonBackendType[self.backend_name.lower()].value
        except KeyError:
            raise AmazonBraketSettingError(f'"{self.backend_name}" is not supported Amazon Braket device.')

        self._check_amazon_braket_availability(AwsDevice(device_arn.value))

        # Amazon Braket device not support the random seed
        return qml.device(
            name=self.device_name,
            device_arn=device_arn.value,
            wires=self.n_qubits,
            shots=self.shots,
            parallel=True,
        )

    def _get_simulator_by_pennylane(self) -> Any:
        """Get simulator by PennlyLane.

        Returns:
            Any: quantum simulator instance
        """
        if self.device_name in AMAZON_BRACKET_DEVICES:
            # Amazon Braket local simulator
            return self._get_amazon_local_simulator_by_pennylane()
        elif self.device_name in AMAZON_BRACKET_REMOTE_DEVICES:
            # Amazon Braket remote simulator
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
        """Get the quantum device instace.

        Returns:
            Any: quantum device instance
        """
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
        return (self.device_name not in IBMQ_REAL_DEVICES) or (self.backend_name in AMAZON_BRAKET_SIMULATOR_BACKENDS)

    def is_remote(self) -> bool:
        """Check the device is a remote device.

        Returns:
            bool: True if the device is a remote device, False otherwise
        """
        return (self.device_name in IBMQ_REAL_DEVICES) or (self.device_name in AMAZON_BRACKET_REMOTE_DEVICES)

    def is_ibmq_device(self) -> bool:
        """Check the device is an IBM Quantum device.

        Returns:
            bool: True if the device is an IBM Quantum device, False otherwise
        """
        return self.device_name in IBMQ_REAL_DEVICES

    def is_amazon_device(self, device_type: Literal["local", "remote", "all"] = "all") -> bool:
        """Check the device is an Amazon Braket device.

        Returns:
            bool: True if the device is an Amazon Braket device, False otherwise
        """
        if device_type == "local":
            return self.device_name in AMAZON_BRACKET_LOCAL_BACKENDS
        elif device_type == "remote":
            return self.device_name in AMAZON_BRACKET_REMOTE_DEVICES
        elif device_type == "all":
            return self.device_name in AMAZON_BRAKET_DEVICES

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

        Raises:
            ValueError: This method is only available for IBM Quantum devices
            IBMQSettingError: Simulator device is not supported
            IBMQSettingError: The real device is not set
        """
        if not self.is_ibmq_device():
            raise ValueError("This method is only available for IBM Quantum devices.")

        if self.is_simulator():
            raise IBMQSettingError(
                f'The device ("{self.device_name}") is a simulator. This method is only available for real devices.'
            )

        if self.real_device is None:
            raise IBMQSettingError(f'The real device ("{self.device_name}") is not set.')

        return self.real_device.service

    def get_backend(self) -> IBMBackend | BackendV2:
        """Get the IBM Quantum real device backend.

        Returns:
            IBMBackend | BackendV2: IBM Quantum real device backend

        Raises:
            ValueError: This method is only available for IBM Quantum devices
            IBMQSettingError: Simulator device is not supported
            IBMQSettingError: The real device is not set
        """
        if not self.is_ibmq_device():
            raise ValueError("This method is only available for IBM Quantum devices.")

        if self.is_simulator():
            raise IBMQSettingError(
                f'The device ("{self.device_name}") is a simulator. This method is only available for real devices.'
            )

        if self.real_device is None:
            raise IBMQSettingError(f'The device ("{self.device_name}") is not set.')

        return self.real_device.backend

    def get_backend_name(self) -> str:
        """Get the real or remote backend name.

        Returns:
            str: backend name
        """
        if self.is_ibmq_device() and self.real_device is not None:
            backend = self.get_backend()
            backend_name = backend.name
        elif self.is_amazon_device() and self.backend_name is not None:
            backend_name = self.backend_name
        else:
            raise ValueError("The backend name is not set.")

        return backend_name

    def _get_ibmq_job_ids(
        self, created_after: Optional[datetime] = None, created_before: Optional[datetime] = None
    ) -> list[str]:
        """Get the IBM Quantum job IDs.

        Args:
            created_after (Optional[datetime]): created datetime of the jobs. If None, start time filter is not applied.
            created_before (Optional[datetime]): finished datetime of the jobs. If None, end time filter is not applied.

        Returns:
            list[str]: IBM Quantum job IDs
        """
        if self.is_simulator():
            raise IBMQSettingError(f'The device ("{self.device_name}") is a simulator.')

        service = self.get_service()
        backend = self.get_backend()
        jobs = service.jobs(backend_name=backend.name, created_after=created_after, created_before=created_before)

        return [job.job_id() for job in jobs]

    def _get_amazon_job_ids(
        self, created_after: Optional[datetime] = None, created_before: Optional[datetime] = None
    ) -> list[str]:
        """Get the Amazon Braket job IDs.
        Amazon Braket API requires "ISO 8601" format for the datetime filter.
        The "created_after" and "created_before" are converted in this method.

        Args:
            created_after (Optional[datetime]): created datetime of the jobs. If None, start time filter is not applied.
            created_before (Optional[datetime]): finished datetime of the jobs. If None, end time filter is not applied.

        Returns:
            list[str]: Amazon Braket job IDs
        """
        braket = boto3.client("braket")
        created_after_utc = created_after.astimezone(timezone.utc).isoformat() if created_after is not None else None
        created_before_utc = created_before.astimezone(timezone.utc).isoformat() if created_before is not None else None

        if created_after_utc is not None and created_before_utc is not None:
            filters = [{"name": "createdAt", "operator": "BETWEEN", "values": [created_after_utc, created_before_utc]}]
        elif created_after_utc is not None and created_before_utc is None:
            filters = [{"name": "createdAt", "operator": "GTE", "values": [created_after_utc]}]
        elif created_after_utc is None and created_before_utc is not None:
            filters = [{"name": "createdAt", "operator": "LTE", "values": [created_before_utc]}]
        else:
            filters = []

        response = braket.search_quantum_tasks(filters=filters)

        return [task["quantumTaskArn"] for task in response.get("quantumTasks", [])]

    def get_job_ids(
        self, created_after: Optional[datetime] = None, created_before: Optional[datetime] = None
    ) -> list[str]:
        """Get the job IDs.

        Returns:
            list[str]: job IDs
        """
        if self.is_ibmq_device():
            return self._get_ibmq_job_ids(created_after=created_after, created_before=created_before)
        elif self.is_amazon_device():
            return self._get_amazon_job_ids(created_after=created_after, created_before=created_before)
        else:
            return []
