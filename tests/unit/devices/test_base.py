import os
from typing import Any, Optional

import pytest
from pennylane.devices.default_qubit import DefaultQubit
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
