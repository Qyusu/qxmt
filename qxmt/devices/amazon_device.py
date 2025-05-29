import os
from datetime import datetime, timezone
from typing import Any, Optional

import boto3
import pennylane as qml
from braket.aws import AwsDevice

from qxmt.constants import AWS_ACCESS_KEY_ID, AWS_DEFAULT_REGION, AWS_SECRET_ACCESS_KEY
from qxmt.devices.amazon import (
    AMAZON_BRAKET_LOCAL_BACKENDS,
    AMAZON_BRAKET_REMOTE_DEVICES,
    AMAZON_BRAKET_SIMULATOR_BACKENDS,
    AMAZON_PROVIDER_NAME,
    AmazonBackendType,
)
from qxmt.devices.base import BaseDevice
from qxmt.exceptions import AmazonBraketSettingError
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class AmazonBraketDevice(BaseDevice):
    """Amazon Braket device implementation for quantum computation.
    This class provides a concrete implementation of the BaseDevice for Amazon Braket.
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
        """Initialize the Amazon Braket device.

        Args:
            platform (str): platform name (ex: pennylane, qulacs, etc.)
            device_name (str): device name provided by the platform (ex: default.qubit, default.tensor, etc.)
            backend_name (Optional[str]): backend name for the Amazon Braket device
            n_qubits (int): number of qubits
            shots (Optional[int]): number of shots for the quantum circuit
            random_seed (Optional[int]): random seed for the quantum device
            logger (Any): logger instance
        """
        super().__init__(platform, device_name, backend_name, n_qubits, shots, random_seed, logger)
        self.aws_access_key_id = None
        self.aws_secret_access_key = None
        self.aws_default_region = None

        if self.device_name in AMAZON_BRAKET_REMOTE_DEVICES:
            self._set_amazon_braket_settings()

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
            raise AmazonBraketSettingError(
                f"Amazon Braket device needs the backend name. Please select in {AMAZON_BRAKET_LOCAL_BACKENDS}."
            )

        if self.backend_name not in AMAZON_BRAKET_LOCAL_BACKENDS:
            raise AmazonBraketSettingError(f'"{self.backend_name}" is not supported Amazon Braket local simulator.')

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

        return qml.device(
            name=self.device_name,
            device_arn=device_arn.value,
            wires=self.n_qubits,
            shots=self.shots,
            parallel=True,
        )

    def get_device(self) -> Any:
        """Get the quantum device instance.

        Returns:
            Any: quantum device instance
        """
        if self.device_name in AMAZON_BRAKET_LOCAL_BACKENDS:
            return self._get_amazon_local_simulator_by_pennylane()
        elif self.device_name in AMAZON_BRAKET_REMOTE_DEVICES:
            return self._get_amazon_remote_device_by_pennylane()
        else:
            raise AmazonBraketSettingError(f'"{self.device_name}" is not supported Amazon Braket device.')

    def is_simulator(self) -> bool:
        """Check if the device is a simulator or real machine.

        Returns:
            bool: True if the device is a simulator, False otherwise
        """
        return self.backend_name in AMAZON_BRAKET_SIMULATOR_BACKENDS

    def is_remote(self) -> bool:
        """Check if the device is a remote device.

        Returns:
            bool: True if the device is a remote device, False otherwise
        """
        return self.device_name in AMAZON_BRAKET_REMOTE_DEVICES

    def get_provider(self) -> str:
        """Get real machine provider name.

        Returns:
            str: provider name
        """
        return AMAZON_PROVIDER_NAME

    def get_backend_name(self) -> str:
        """Get the real or remote backend name.

        Returns:
            str: backend name
        """
        if self.backend_name is None:
            raise ValueError("The backend name is not set.")
        return self.backend_name

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
