import os

import pytest
from pytest_mock import MockFixture

from qxmt.constants import IBMQ_API_KEY
from qxmt.devices.ibmq import IBMQ_PROVIDER_NAME
from qxmt.devices.ibmq_device import IBMQDevice
from qxmt.exceptions import IBMQSettingError


@pytest.fixture
def ibmq_real_device(mocker: MockFixture) -> IBMQDevice:
    mocker.patch.dict(os.environ, {IBMQ_API_KEY: "test_key"})
    return IBMQDevice(
        platform="pennylane",
        device_name="qiskit.remote",
        backend_name=None,
        n_qubits=5,
        shots=1024,
    )


class TestIBMQProperty:
    def test_get_device(self, mocker: MockFixture, ibmq_real_device: IBMQDevice) -> None:
        mock_device = mocker.Mock()

        def fake_set_real_device():
            ibmq_real_device.real_device = mock_device

        mocker.patch.object(ibmq_real_device, "_set_ibmq_real_device_by_pennylane", side_effect=fake_set_real_device)

        device = ibmq_real_device.get_device()
        assert device is not None

    def test_is_simulator(self, ibmq_real_device: IBMQDevice) -> None:
        assert ibmq_real_device.is_simulator() is False

    def test_is_remote(self, ibmq_real_device: IBMQDevice) -> None:
        assert ibmq_real_device.is_remote() is True

    def test_get_provider(self, ibmq_real_device: IBMQDevice) -> None:
        assert ibmq_real_device.get_provider() == IBMQ_PROVIDER_NAME

    def test_get_service(self, mocker: MockFixture, ibmq_real_device: IBMQDevice) -> None:
        mock_service = mocker.Mock()
        mock_backend = mocker.Mock()
        mock_backend.name = "ibmq_test_backend"
        mock_backend.num_qubits = 5
        mock_service.least_busy.return_value = mock_backend
        mock_real_device = mocker.Mock()
        mock_real_device.service = mock_service
        mock_real_device.backend = mock_backend
        ibmq_real_device.real_device = mock_real_device
        service = ibmq_real_device.get_service()
        assert service == mock_service

    def test_get_backend(self, mocker: MockFixture, ibmq_real_device: IBMQDevice) -> None:
        mock_service = mocker.Mock()
        mock_backend = mocker.Mock()
        mock_service.least_busy.return_value = mock_backend
        mock_real_device = mocker.Mock()
        mock_real_device.backend = mock_backend
        # Error case: before setting real_device
        with pytest.raises(IBMQSettingError) as exc_info:
            ibmq_real_device.get_backend()
        assert 'The real device ("qiskit.remote") is not set.' in str(exc_info.value)
        # Pass case: after setting real_device
        ibmq_real_device.real_device = mock_real_device
        backend = ibmq_real_device.get_backend()
        assert backend == mock_backend

    def test_get_backend_name(self, mocker: MockFixture, ibmq_real_device: IBMQDevice) -> None:
        mock_backend = mocker.Mock()
        mock_backend.name = "ibmq_test_backend"
        mock_real_device = mocker.Mock()
        mock_real_device.backend = mock_backend
        ibmq_real_device.real_device = mock_real_device
        backend_name = ibmq_real_device.get_backend_name()
        assert backend_name == "ibmq_test_backend"

    def test_get_job_ids(self, mocker: MockFixture, ibmq_real_device: IBMQDevice) -> None:
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
