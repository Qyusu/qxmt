from typing import Any, Optional

import pytest
from pennylane.devices.default_qubit import DefaultQubit

from qxmt.devices.base import BaseDevice
from qxmt.exceptions import InvalidPlatformError


class TestBaseDevice:
    @pytest.mark.parametrize(
        "platform, name, n_qubits, shots, random_seed",
        [
            pytest.param("pennylane", "default.qubit", 2, None, None, id="state vector mode"),
            pytest.param("pennylane", "default.qubit", 2, 100, 42, id="sampling mode"),
        ],
    )
    def test__init__(
        self, platform: str, name: str, n_qubits: int, shots: Optional[int], random_seed: Optional[int]
    ) -> None:
        device = BaseDevice(platform=platform, name=name, n_qubits=n_qubits, shots=shots, random_seed=random_seed)
        assert device.platform == platform
        assert device.name == name
        assert device.n_qubits == n_qubits
        if shots is None:
            assert device.shots is shots
        else:
            assert device.shots == shots
        if random_seed is None:
            assert device.random_seed is random_seed
        else:
            assert device.random_seed == random_seed

    @pytest.mark.parametrize(
        "platform, name, n_qubits, shots, random_seed, expected",
        [
            pytest.param("pennylane", "default.qubit", 2, None, None, DefaultQubit, id="pennylane simulator"),
            pytest.param("not_supported", "default.qubit", 2, 100, 42, None, id="not supported platform"),
        ],
    )
    def test_get_simulator(
        self, platform: str, name: str, n_qubits: int, shots: Optional[int], random_seed: Optional[int], expected: Any
    ) -> None:
        if expected is None:
            with pytest.raises(InvalidPlatformError):
                _ = BaseDevice(
                    platform=platform, name=name, n_qubits=n_qubits, shots=shots, random_seed=random_seed
                ).get_simulator()
        else:
            simulator = BaseDevice(
                platform=platform, name=name, n_qubits=n_qubits, shots=shots, random_seed=random_seed
            ).get_simulator()
            assert isinstance(simulator, expected)

    def test_get_real_machine(self) -> None:
        device = BaseDevice(platform="pennylane", name="default.qubit", n_qubits=2, shots=None, random_seed=None)
        with pytest.raises(NotImplementedError):
            device.get_real_machine()
