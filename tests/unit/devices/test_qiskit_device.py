from typing import Any

import pytest

pytest.importorskip("qiskit_aer")

from qxmt.devices.qiskit_device import QiskitDevice


@pytest.fixture
def qiskit_device() -> QiskitDevice:
    return QiskitDevice(
        platform="qiskit",
        device_name="automatic",
        backend_name=None,
        n_qubits=2,
        shots=100,
        device_options=None,
    )


class TestQiskitDevice:
    def test_get_device(self, qiskit_device: QiskitDevice) -> None:
        device = qiskit_device.get_device()
        assert device is not None
        assert device.options.method == "automatic"

    def test_is_simulator(self, qiskit_device: QiskitDevice) -> None:
        assert qiskit_device.is_simulator() is True

    def test_is_remote(self, qiskit_device: QiskitDevice) -> None:
        assert qiskit_device.is_remote() is False

    def test_get_provider(self, qiskit_device: QiskitDevice) -> None:
        assert qiskit_device.get_provider() == ""

    def test_get_backend_name(self, qiskit_device: QiskitDevice) -> None:
        assert qiskit_device.get_backend_name() == ""

    def test_get_job_ids(self, qiskit_device: QiskitDevice) -> None:
        assert qiskit_device.get_job_ids() == []

    def test_device_options_passed_to_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_kwargs: dict[str, Any] = {}

        class FakeAerSimulator:
            def __init__(self, **kwargs: Any) -> None:
                captured_kwargs.update(kwargs)

        monkeypatch.setattr("qiskit_aer.AerSimulator", FakeAerSimulator)

        device = QiskitDevice(
            platform="qiskit",
            device_name="statevector",
            backend_name=None,
            n_qubits=2,
            shots=100,
            device_options={"seed_simulator": 42},
        )
        device_handle = device.get_device()

        assert device_handle is not None
        assert captured_kwargs["method"] == "statevector"
        assert captured_kwargs["seed_simulator"] == 42
        assert "shots" not in captured_kwargs

    def test_invalid_device_option_key(self) -> None:
        with pytest.raises(ValueError):
            QiskitDevice(
                platform="qiskit",
                device_name="automatic",
                backend_name=None,
                n_qubits=2,
                shots=100,
                device_options={"method": "statevector"},
            )
