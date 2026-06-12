from typing import Any

import pytest

from qxmt.devices.pennylane_device import PennyLaneDevice


@pytest.fixture
def simulator_cpu_device() -> PennyLaneDevice:
    return PennyLaneDevice(
        platform="pennylane",
        device_name="default.qubit",
        backend_name=None,
        n_qubits=2,
        shots=100,
        device_options=None,
    )


class TestPennyLaneCPUDevice:
    def test_get_device(self, simulator_cpu_device: PennyLaneDevice) -> None:
        device = simulator_cpu_device.get_device()
        assert device is not None

    def test_is_simulator(self, simulator_cpu_device: PennyLaneDevice) -> None:
        assert simulator_cpu_device.is_simulator() is True

    def test_is_remote(self, simulator_cpu_device: PennyLaneDevice) -> None:
        assert simulator_cpu_device.is_remote() is False

    def test_get_provider(self, simulator_cpu_device: PennyLaneDevice) -> None:
        assert simulator_cpu_device.get_provider() == ""

    def test_get_backend_name(self, simulator_cpu_device: PennyLaneDevice) -> None:
        assert simulator_cpu_device.get_backend_name() == ""

    def test_get_job_ids(self, simulator_cpu_device: PennyLaneDevice) -> None:
        assert simulator_cpu_device.get_job_ids() == []

    def test_device_options_passed_to_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_kwargs: dict[str, Any] = {}

        def fake_device(*args: Any, **kwargs: Any) -> object:
            captured_kwargs.update(kwargs)
            return object()

        monkeypatch.setattr("qxmt.devices.pennylane_device.qml.device", fake_device)

        device = PennyLaneDevice(
            platform="pennylane",
            device_name="default.qubit",
            backend_name=None,
            n_qubits=2,
            shots=100,
            device_options={"seed": 42},
        )
        device_handle = device.get_device()

        assert device_handle is not None
        assert captured_kwargs["seed"] == 42
        assert "shots" not in captured_kwargs

    def test_invalid_device_option_key(self) -> None:
        with pytest.raises(ValueError):
            PennyLaneDevice(
                platform="pennylane",
                device_name="default.qubit",
                backend_name=None,
                n_qubits=2,
                shots=100,
                device_options={"wires": 3, "shots": 200},
            )

    def test_get_device_with_unsupported_option(self) -> None:
        device = PennyLaneDevice(
            platform="pennylane",
            device_name="null.qubit",
            backend_name=None,
            n_qubits=2,
            shots=100,
            device_options={"seed": 42},
        )
        with pytest.raises(TypeError):
            device.get_device()
