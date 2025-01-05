import os
from datetime import datetime
from typing import Optional

import pytest
from pytest_mock import MockFixture

from qxmt.devices.base import BaseDevice
from qxmt.exceptions import IBMQSettingError, InvalidPlatformError


@pytest.mark.parametrize(
    "platform, device_name, backend_name, n_qubits, shots, random_seed",
    [
        pytest.param("pennylane", "default.qubit", None, 2, None, None, id="state vector mode"),
        pytest.param("pennylane", "default.qubit", None, 2, 100, 42, id="sampling mode"),
    ],
)
def test__init__(
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


def test_invalid_platform_error() -> None:
    with pytest.raises(InvalidPlatformError):
        BaseDevice(
            platform="unknown_platform",
            device_name="default.qubit",
            backend_name=None,
            n_qubits=2,
            shots=100,
        )


def test_ibmq_setting_error(mocker: MockFixture) -> None:
    # IBMQ_API_KEY is not set
    mocker.patch.dict(os.environ, {}, clear=True)
    with pytest.raises(IBMQSettingError):
        BaseDevice(
            platform="pennylane",
            device_name="qiskit.remote",
            backend_name=None,
            n_qubits=2,
            shots=100,
        )


def test_is_simulator(mocker: MockFixture):
    # Simulator
    mocker.patch("pennylane.device")
    device_sim = BaseDevice(
        platform="pennylane",
        device_name="default.qubit",
        backend_name=None,
        n_qubits=2,
        shots=100,
    )
    assert device_sim.is_simulator() is True

    # Real Machine
    mocker.patch.dict(os.environ, {"IBMQ_API_KEY": "test_key"})
    mock_service = mocker.Mock()
    mock_backend = mocker.Mock()
    mock_service.least_busy.return_value = mock_backend
    mocker.patch("qxmt.devices.base.QiskitRuntimeService", return_value=mock_service)
    mocker.patch("pennylane.device")
    device_real = BaseDevice(
        platform="pennylane",
        device_name="qiskit.remote",
        backend_name=None,
        n_qubits=5,
        shots=1024,
    )
    assert device_real.is_simulator() is False


def test_get_provider(mocker: MockFixture):
    # Simulator
    mocker.patch("pennylane.device")
    device_sim = BaseDevice(
        platform="pennylane",
        device_name="default.qubit",
        backend_name=None,
        n_qubits=2,
        shots=100,
    )
    assert device_sim.get_provider() == ""

    # Real Machine
    mocker.patch.dict(os.environ, {"IBMQ_API_KEY": "test_key"})
    mock_service = mocker.Mock()
    mock_backend = mocker.Mock()
    mock_service.least_busy.return_value = mock_backend
    mocker.patch("qxmt.devices.base.QiskitRuntimeService", return_value=mock_service)
    mocker.patch("pennylane.device")
    device_real = BaseDevice(
        platform="pennylane",
        device_name="qiskit.remote",
        backend_name=None,
        n_qubits=5,
        shots=1024,
    )
    assert device_real.get_provider() == "IBM"


def test_get_service_real_device(mocker: MockFixture):
    mocker.patch.dict(os.environ, {"IBMQ_API_KEY": "test_key"})
    mock_service = mocker.Mock()
    mock_backend = mocker.Mock()
    mock_service.least_busy.return_value = mock_backend
    mocker.patch("qxmt.devices.base.QiskitRuntimeService", return_value=mock_service)

    mock_qml_device = mocker.Mock()
    mock_qml_device.backend = mock_backend
    mock_qml_device.service = mock_service
    mocker.patch("pennylane.device", return_value=mock_qml_device)

    device = BaseDevice(
        platform="pennylane",
        device_name="qiskit.remote",
        backend_name=None,
        n_qubits=5,
        shots=1024,
    )

    service = device.get_service()
    assert service == mock_service


def test_get_service_simulator(mocker: MockFixture):
    mocker.patch("pennylane.device")
    device = BaseDevice(
        platform="pennylane",
        device_name="default.qubit",
        backend_name=None,
        n_qubits=2,
        shots=100,
    )

    with pytest.raises(IBMQSettingError) as exc_info:
        device.get_service()

    assert 'The device ("default.qubit") is a simulator.' in str(exc_info.value)


def test_get_backend_real_device(mocker: MockFixture):
    mocker.patch.dict(os.environ, {"IBMQ_API_KEY": "test_key"})
    mock_service = mocker.Mock()
    mock_backend = mocker.Mock()
    mock_service.least_busy.return_value = mock_backend
    mocker.patch("qxmt.devices.base.QiskitRuntimeService", return_value=mock_service)

    mock_qml_device = mocker.Mock()
    mock_qml_device.backend = mock_backend
    mock_qml_device.service = mock_service
    mocker.patch("pennylane.device", return_value=mock_qml_device)

    device = BaseDevice(
        platform="pennylane",
        device_name="qiskit.remote",
        backend_name=None,
        n_qubits=5,
        shots=1024,
    )

    backend = device.get_backend()
    assert backend == mock_backend


def test_get_backend_simulator(mocker: MockFixture):
    mocker.patch("pennylane.device")
    device = BaseDevice(
        platform="pennylane",
        device_name="default.qubit",
        backend_name=None,
        n_qubits=2,
        shots=100,
    )

    with pytest.raises(IBMQSettingError) as exc_info:
        device.get_backend()

    assert 'The device ("default.qubit") is a simulator.' in str(exc_info.value)


def test_get_backend_name_real_device(mocker: MockFixture):
    mocker.patch.dict(os.environ, {"IBMQ_API_KEY": "test_key"})
    mock_service = mocker.Mock()
    mock_backend = mocker.Mock()
    mock_backend.name = "ibmq_test_backend"
    mock_service.least_busy.return_value = mock_backend
    mocker.patch("qxmt.devices.base.QiskitRuntimeService", return_value=mock_service)

    mock_qml_device = mocker.Mock()
    mock_qml_device.backend = mock_backend
    mock_qml_device.service = mock_service
    mocker.patch("pennylane.device", return_value=mock_qml_device)

    device = BaseDevice(
        platform="pennylane",
        device_name="qiskit.remote",
        backend_name=None,
        n_qubits=5,
        shots=1024,
    )

    backend_name = device.get_backend_name()
    assert backend_name == "ibmq_test_backend"


def test_get_ibmq_job_ids_real_device(mocker: MockFixture):
    mocker.patch.dict(os.environ, {"IBMQ_API_KEY": "test_key"})
    mock_service = mocker.Mock()
    mock_backend = mocker.Mock()
    mock_backend.name = "ibmq_test_backend"
    mock_backend.num_qubits = 5
    mock_service.least_busy.return_value = mock_backend
    mock_service.jobs.return_value = [
        mocker.Mock(job_id=lambda: "job1"),
        mocker.Mock(job_id=lambda: "job2"),
    ]
    mocker.patch("qxmt.devices.base.QiskitRuntimeService", return_value=mock_service)

    mock_qml_device = mocker.Mock()
    mock_qml_device.service = mock_service
    mocker.patch("pennylane.device", return_value=mock_qml_device)

    device = BaseDevice(
        platform="pennylane",
        device_name="qiskit.remote",
        backend_name=None,
        n_qubits=5,
        shots=1024,
    )

    job_ids = device.get_ibmq_job_ids(created_after=None, created_before=None)

    assert job_ids == ["job1", "job2"]


def test_get_ibmq_job_ids_simulator(mocker: MockFixture):
    mocker.patch("pennylane.device")
    device = BaseDevice(
        platform="pennylane",
        device_name="default.qubit",
        backend_name=None,
        n_qubits=2,
        shots=100,
    )

    with pytest.raises(IBMQSettingError) as exc_info:
        device.get_ibmq_job_ids()

    assert 'The device ("default.qubit") is a simulator.' in str(exc_info.value)


def test_call_device(mocker: MockFixture):
    mock_qml_device = mocker.Mock()
    mocker.patch("pennylane.device", return_value=mock_qml_device)

    device = BaseDevice(
        platform="pennylane",
        device_name="default.qubit",
        backend_name=None,
        n_qubits=2,
        shots=100,
    )

    returned_device = device()
    assert returned_device == mock_qml_device
