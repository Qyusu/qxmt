import os

import pytest
from pytest_mock import MockFixture

from qxmt.constants import AWS_ACCESS_KEY_ID, AWS_DEFAULT_REGION, AWS_SECRET_ACCESS_KEY
from qxmt.devices.amazon_device import AmazonBraketDevice
from qxmt.exceptions import AmazonBraketSettingError


@pytest.fixture
def amazon_local_simulator_device() -> AmazonBraketDevice:
    return AmazonBraketDevice(
        platform="pennylane",
        device_name="braket.local.qubit",
        backend_name="braket_sv",
        n_qubits=5,
        shots=1024,
    )


@pytest.fixture
def amazon_remote_simulator_device() -> AmazonBraketDevice:
    return AmazonBraketDevice(
        platform="pennylane",
        device_name="braket.local.qubit",
        backend_name="sv1",
        n_qubits=5,
        shots=1024,
    )


@pytest.fixture
def amazon_remote_real_device(mocker: MockFixture) -> AmazonBraketDevice:
    mocker.patch.dict(os.environ, {AWS_ACCESS_KEY_ID: "test_key"})
    mocker.patch.dict(os.environ, {AWS_SECRET_ACCESS_KEY: "test_secret"})
    mocker.patch.dict(os.environ, {AWS_DEFAULT_REGION: "us-west-2"})
    return AmazonBraketDevice(
        platform="pennylane",
        device_name="braket.local.qubit",
        backend_name="ionq",
        n_qubits=5,
        shots=1024,
    )


class TestAmazonProperty:
    def test_get_amazon_local_simulator_by_pennylane(self, amazon_local_simulator_device: AmazonBraketDevice) -> None:
        device_with_unsupported_backend = AmazonBraketDevice(
            platform="pennylane",
            device_name="braket.local.qubit",
            backend_name="not_supported",
            n_qubits=5,
            shots=1024,
        )
        # Error case: backend_name is not supported
        with pytest.raises(AmazonBraketSettingError) as exc_info:
            device_with_unsupported_backend._get_amazon_local_simulator_by_pennylane()
        assert '"not_supported" is not supported Amazon Braket local simulator.' in str(exc_info.value)
        # Pass case: backend_name is None. Set the default backend name.
        device = AmazonBraketDevice(
            platform="pennylane",
            device_name="braket.local.qubit",
            backend_name=None,
            n_qubits=5,
            shots=1024,
        )
        with pytest.raises(AmazonBraketSettingError):
            # backend_name=NoneはAmazonBraketDeviceの仕様上エラーになる
            device._get_amazon_local_simulator_by_pennylane()
        # Pass case: backend_name is supported
        assert amazon_local_simulator_device.backend_name == "braket_sv"

    def test_get_amazon_remote_device_by_pennylane(
        self, amazon_remote_simulator_device: AmazonBraketDevice, amazon_remote_real_device: AmazonBraketDevice
    ) -> None:
        device_with_no_backend = AmazonBraketDevice(
            platform="pennylane",
            device_name="braket.aws.qubit",
            backend_name=None,
            n_qubits=5,
            shots=1024,
        )
        # Error case: backend_name is None
        with pytest.raises(AmazonBraketSettingError) as exc_info:
            device_with_no_backend._get_amazon_remote_device_by_pennylane()
        assert "Amazon Braket device needs the backend name." in str(exc_info.value)
        device_with_unsupported_backend = AmazonBraketDevice(
            platform="pennylane",
            device_name="braket.aws.qubit",
            backend_name="not_supported",
            n_qubits=5,
            shots=1024,
        )
        # Error case: backend_name is not supported
        with pytest.raises(AmazonBraketSettingError) as exc_info:
            device_with_unsupported_backend._get_amazon_remote_device_by_pennylane()
        assert '"not_supported" is not supported Amazon Braket device.' in str(exc_info.value)
        # Pass case: backend_name is supported (simulator)
        assert amazon_remote_simulator_device.backend_name == "sv1"
        # Pass case: backend_name is supported (real device)
        assert amazon_remote_real_device.backend_name == "ionq"

    def test_get_backend_name(
        self,
        amazon_local_simulator_device: AmazonBraketDevice,
        amazon_remote_simulator_device: AmazonBraketDevice,
        amazon_remote_real_device: AmazonBraketDevice,
    ) -> None:
        local_backend_name = amazon_local_simulator_device.get_backend_name()
        assert local_backend_name == "braket_sv"
        simulator_backend_name = amazon_remote_simulator_device.get_backend_name()
        assert simulator_backend_name == "sv1"
        real_backend_name = amazon_remote_real_device.get_backend_name()
        assert real_backend_name == "ionq"

    def test_get_job_ids(self, mocker: MockFixture, amazon_remote_real_device: AmazonBraketDevice) -> None:
        from datetime import datetime, timezone

        mock_boto3_client = mocker.patch("boto3.client")
        mock_braket = mocker.Mock()
        mock_boto3_client.return_value = mock_braket
        job_1 = "arn:aws:braket:task/1234"
        job_2 = "arn:aws:braket:task/5678"
        mock_braket.search_quantum_tasks.return_value = {
            "quantumTasks": [
                {"quantumTaskArn": job_1},
                {"quantumTaskArn": job_2},
            ]
        }
        created_after = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        created_before = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        job_ids = amazon_remote_real_device.get_job_ids(created_after=created_after, created_before=created_before)
        assert job_ids == [job_1, job_2]
