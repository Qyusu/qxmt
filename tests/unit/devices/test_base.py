import os
from typing import Optional

import pytest
from pytest_mock import MockFixture

from qxmt.devices.base import BaseDevice
from qxmt.exceptions import AmazonBraketSettingError, IBMQSettingError


@pytest.fixture
def ibmq_real_device(mocker: MockFixture) -> BaseDevice:
    mocker.patch.dict(os.environ, {"IBMQ_API_KEY": "test_key"})
    return BaseDevice(
        platform="pennylane",
        device_name="qiskit.remote",
        backend_name=None,
        n_qubits=5,
        shots=1024,
    )


@pytest.fixture
def amazon_simulator_device() -> BaseDevice:
    return BaseDevice(
        platform="pennylane",
        device_name="braket.local.qubit",
        backend_name=None,
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
        self, simulator_device: BaseDevice, ibmq_real_device: BaseDevice, amazon_simulator_device: BaseDevice
    ) -> None:
        # Simulator
        assert simulator_device.get_provider() == ""

        # IBMQ Real Machine
        assert ibmq_real_device.get_provider() == "IBM_Quantum"

        # Amazon Braket Simulator
        assert amazon_simulator_device.get_provider() == "Amazon_Braket"


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
        with pytest.raises(IBMQSettingError) as exc_info:
            simulator_device.get_service()

        assert 'The device ("default.qubit") is a simulator.' in str(exc_info.value)

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
        with pytest.raises(IBMQSettingError) as exc_info:
            simulator_device.get_backend()

        assert 'The device ("default.qubit") is a simulator.' in str(exc_info.value)

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

    def test_real_device_case(self, mocker: MockFixture, ibmq_real_device: BaseDevice) -> None:
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

        job_ids = ibmq_real_device.get_ibmq_job_ids(created_after=None, created_before=None)
        assert job_ids == ["job1", "job2"]

    def test_simulator_case(self, simulator_device: BaseDevice) -> None:
        with pytest.raises(IBMQSettingError) as exc_info:
            simulator_device.get_ibmq_job_ids()

        assert 'The device ("default.qubit") is a simulator.' in str(exc_info.value)


class TestAmazonProperty:
    def test_get_amazon_local_simulator_by_pennylane(self) -> None:
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

        # Pass case: backend_name is None
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
        device = BaseDevice(
            platform="pennylane",
            device_name="braket.local.qubit",
            backend_name="braket_sv",
            n_qubits=5,
            shots=1024,
        )
        device._get_amazon_local_simulator_by_pennylane()
        assert device.backend_name == "braket_sv"

    def test_get_amazon_remote_device_by_pennylane(self) -> None:
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
