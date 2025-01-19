import os
from typing import Optional

import pytest
from pytest_mock import MockFixture

from qxmt.constants import (
    AWS_ACCESS_KEY_ID,
    AWS_DEFAULT_REGION,
    AWS_SECRET_ACCESS_KEY,
    IBMQ_API_KEY,
)
from qxmt.devices.base import BaseDevice
from qxmt.exceptions import AmazonBraketSettingError, IBMQSettingError


@pytest.fixture
def ibmq_real_device(mocker: MockFixture) -> BaseDevice:
    mocker.patch.dict(os.environ, {IBMQ_API_KEY: "test_key"})
    return BaseDevice(
        platform="pennylane",
        device_name="qiskit.remote",
        backend_name=None,
        n_qubits=5,
        shots=1024,
    )


@pytest.fixture
def amazon_local_simulator_device() -> BaseDevice:
    return BaseDevice(
        platform="pennylane",
        device_name="braket.local.qubit",
        backend_name="braket_sv",
        n_qubits=5,
        shots=1024,
    )


@pytest.fixture
def amazon_remote_simulator_device() -> BaseDevice:
    return BaseDevice(
        platform="pennylane",
        device_name="braket.local.qubit",
        backend_name="sv1",
        n_qubits=5,
        shots=1024,
    )


@pytest.fixture
def amazon_remote_real_device(mocker: MockFixture) -> BaseDevice:
    mocker.patch.dict(os.environ, {AWS_ACCESS_KEY_ID: "test_key"})
    mocker.patch.dict(os.environ, {AWS_SECRET_ACCESS_KEY: "test_secret"})
    mocker.patch.dict(os.environ, {AWS_DEFAULT_REGION: "us-west-2"})
    return BaseDevice(
        platform="pennylane",
        device_name="braket.local.qubit",
        backend_name="ionq",
        n_qubits=5,
        shots=1024,
    )


@pytest.fixture
def simulator_device() -> BaseDevice:
    return BaseDevice(
        platform="pennylane",
        device_name="default.qubit",
        backend_name=None,
        n_qubits=2,
        shots=100,
    )


class TestBaseDeviceMethod:
    @pytest.mark.parametrize(
        "platform, device_name, backend_name, n_qubits, shots, random_seed",
        [
            pytest.param("pennylane", "default.qubit", None, 2, None, None, id="state vector mode"),
            pytest.param("pennylane", "default.qubit", None, 2, 100, 42, id="sampling mode"),
        ],
    )
    def test__init__(
        self,
        platform: str,
        device_name: str,
        backend_name: str,
        n_qubits: int,
        shots: Optional[int],
        random_seed: Optional[int],
    ) -> None:
        device = BaseDevice(
            platform=platform,
            device_name=device_name,
            backend_name=backend_name,
            n_qubits=n_qubits,
            shots=shots,
            random_seed=random_seed,
        )
        assert device.platform == platform
        assert device.device_name == device_name
        assert device.n_qubits == n_qubits

        if shots is None:
            assert device.shots is shots
        else:
            assert device.shots == shots

        if random_seed is None:
            assert device.random_seed is random_seed
        else:
            assert device.random_seed == random_seed

        self.real_device = None

    def test_get_device(self, mocker: MockFixture) -> None:
        # [TODO]: Implement this test
        pass

    def test_is_simulator(self, simulator_device: BaseDevice, ibmq_real_device: BaseDevice) -> None:
        # Simulator
        assert simulator_device.is_simulator() is True

        # Real Machine
        assert ibmq_real_device.is_simulator() is False

    def test_get_provider(
        self,
        simulator_device: BaseDevice,
        ibmq_real_device: BaseDevice,
        amazon_local_simulator_device: BaseDevice,
        amazon_remote_simulator_device: BaseDevice,
        amazon_remote_real_device: BaseDevice,
    ) -> None:
        # Pennylane Original Simulator
        assert simulator_device.get_provider() == ""

        # IBMQ
        assert ibmq_real_device.get_provider() == "IBM_Quantum"

        # Amazon Braket
        assert amazon_local_simulator_device.get_provider() == "Amazon_Braket"
        assert amazon_remote_simulator_device.get_provider() == "Amazon_Braket"
        assert amazon_remote_real_device.get_provider() == "Amazon_Braket"


class TestIBMQProperty:
    def test_get_service_real_device(self, mocker: MockFixture, ibmq_real_device: BaseDevice) -> None:
        mock_service = mocker.Mock()
        mock_backend = mocker.Mock()
        mock_backend.name = "ibmq_test_backend"
        mock_backend.num_qubits = 5
        mock_service.least_busy.return_value = mock_backend
        mock_real_device = mocker.Mock()
        mock_real_device.service = mock_service
        mock_real_device.backend = mock_backend

        # set mock on real_device
        ibmq_real_device.real_device = mock_real_device

        service = ibmq_real_device.get_service()
        assert service == mock_service

    def test_get_service_simulator(self, simulator_device: BaseDevice) -> None:
        with pytest.raises(ValueError) as exc_info:
            simulator_device.get_backend()

        assert "This method is only available for IBM Quantum devices." in str(exc_info.value)

    def test_get_backend_real_device(self, mocker: MockFixture, ibmq_real_device: BaseDevice) -> None:
        # mock for real device
        mock_service = mocker.Mock()
        mock_backend = mocker.Mock()
        mock_service.least_busy.return_value = mock_backend
        mock_real_device = mocker.Mock()
        mock_real_device.backend = mock_backend

        # Error case: befor setting real_device
        with pytest.raises(IBMQSettingError) as exc_info:
            ibmq_real_device.get_backend()
        assert 'The device ("qiskit.remote") is not set.' in str(exc_info.value)

        # Pass case: after setting real_device
        ibmq_real_device.real_device = mock_real_device
        backend = ibmq_real_device.get_backend()
        assert backend == mock_backend

    def test_get_backend_simulator(self, simulator_device: BaseDevice) -> None:
        with pytest.raises(ValueError) as exc_info:
            simulator_device.get_backend()

        assert "This method is only available for IBM Quantum devices." in str(exc_info.value)

    def test_get_backend_name_real_device(self, mocker: MockFixture, ibmq_real_device: BaseDevice) -> None:
        mock_service = mocker.Mock()
        mock_backend = mocker.Mock()
        mock_backend.name = "ibmq_test_backend"
        mock_service.least_busy.return_value = mock_backend
        mock_real_device = mocker.Mock()
        mock_real_device.backend = mock_backend

        ibmq_real_device.real_device = mock_real_device
        backend_name = ibmq_real_device.get_backend_name()
        assert backend_name == "ibmq_test_backend"

    def test_get_job_ids(self, mocker: MockFixture, ibmq_real_device: BaseDevice) -> None:
        mock_service = mocker.Mock()
        mock_backend = mocker.Mock()
        mock_backend.name = "ibmq_test_backend"
        mock_backend.num_qubits = 5
        mock_service.least_busy.return_value = mock_backend
        mock_real_device = mocker.Mock()
        mock_real_device.service = mock_service
        mock_real_device.backend = mock_backend
        mock_service.jobs.return_value = [
            mocker.Mock(job_id=lambda: "job1"),
            mocker.Mock(job_id=lambda: "job2"),
        ]

        ibmq_real_device.real_device = mock_real_device

        job_ids = ibmq_real_device.get_job_ids(created_after=None, created_before=None)
        assert job_ids == ["job1", "job2"]


class TestAmazonProperty:
    def test_get_amazon_local_simulator_by_pennylane(self, amazon_local_simulator_device: BaseDevice) -> None:
        # Error case: backend_name is not supported
        with pytest.raises(AmazonBraketSettingError) as exc_info:
            BaseDevice(
                platform="pennylane",
                device_name="braket.local.qubit",
                backend_name="not_supported",
                n_qubits=5,
                shots=1024,
            )._get_amazon_local_simulator_by_pennylane()

        assert '"not_supported" is not supported Amazon Braket local simulator.' in str(exc_info.value)

        # Pass case: backend_name is None. Set the default backend name.
        device = BaseDevice(
            platform="pennylane",
            device_name="braket.local.qubit",
            backend_name=None,
            n_qubits=5,
            shots=1024,
        )
        device._get_amazon_local_simulator_by_pennylane()
        assert device.backend_name == "braket_sv"

        # Pass case: backend_name is supported
        assert amazon_local_simulator_device.backend_name == "braket_sv"

    def test_get_amazon_remote_device_by_pennylane(
        self, amazon_remote_simulator_device: BaseDevice, amazon_remote_real_device: BaseDevice
    ) -> None:
        # Error case: backend_name is None
        with pytest.raises(AmazonBraketSettingError) as exc_info:
            BaseDevice(
                platform="pennylane",
                device_name="braket.aws.qubit",
                backend_name=None,
                n_qubits=5,
                shots=1024,
            )._get_amazon_remote_device_by_pennylane()

        assert "Amazon Braket device needs the backend name." in str(exc_info.value)

        # Error case: backend_name is not supported
        with pytest.raises(AmazonBraketSettingError) as exc_info:
            BaseDevice(
                platform="pennylane",
                device_name="braket.aws.qubit",
                backend_name="not_supported",
                n_qubits=5,
                shots=1024,
            )._get_amazon_remote_device_by_pennylane()

        assert '"not_supported" is not supported Amazon Braket device.' in str(exc_info.value)

        # Pass case: backend_name is supported (simulator)
        assert amazon_remote_simulator_device.backend_name == "sv1"

        # Pass case: backend_name is supported (real device)
        assert amazon_remote_real_device.backend_name == "ionq"

    def test_get_backend_name(
        self,
        amazon_local_simulator_device: BaseDevice,
        amazon_remote_simulator_device: BaseDevice,
        amazon_remote_real_device: BaseDevice,
    ) -> None:
        local_backend_name = amazon_local_simulator_device.get_backend_name()
        assert local_backend_name == "braket_sv"

        simulator_backend_name = amazon_remote_simulator_device.get_backend_name()
        assert simulator_backend_name == "sv1"

        real_backend_name = amazon_remote_real_device.get_backend_name()
        assert real_backend_name == "ionq"

    def test_get_job_ids(self, mocker: MockFixture, amazon_remote_real_device: BaseDevice) -> None:
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
