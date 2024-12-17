from typing import Any, Optional

import pytest
from pennylane.devices.default_qubit import DefaultQubit

from qxmt.devices.base import BaseDevice
from qxmt.exceptions import InvalidPlatformError


class TestBaseDevice:
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

    @pytest.mark.parametrize(
        "platform, device_name, backend_name, n_qubits, shots, random_seed, expected",
        [
            pytest.param("pennylane", "default.qubit", None, 2, None, None, DefaultQubit, id="pennylane simulator"),
            pytest.param("not_supported", "default.qubit", None, 2, 100, 42, None, id="not supported platform"),
        ],
    )
    def test_set_simulator_by_pennylane(
        self,
        platform: str,
        device_name: str,
        backend_name: str,
        n_qubits: int,
        shots: Optional[int],
        random_seed: Optional[int],
        expected: Any,
    ) -> None:
        pass
        # if expected is None:
        #     with pytest.raises(InvalidPlatformError):
        #         _ = BaseDevice(
        #             platform=platform,
        #             device_name=device_name,
        #             backend_name=backend_name,
        #             n_qubits=n_qubits,
        #             shots=shots,
        #             random_seed=random_seed,
        #         )._set_simulator_by_pennylane()
        # else:
        #     simulator = BaseDevice(
        #         platform=platform,
        #         device_name=device_name,
        #         backend_name=backend_name,
        #         n_qubits=n_qubits,
        #         shots=shots,
        #         random_seed=random_seed,
        #     )._set_simulator_by_pennylane()
        #     assert isinstance(simulator, expected)

    def test__set_ibmq_real_device_by_pennylane(self) -> None:
        pass
