import pytest

from qxmt.devices.qulacs_device import QulacsDevice


@pytest.fixture
def qulacs_device() -> QulacsDevice:
    return QulacsDevice(
        platform="qulacs",
        device_name="cpu.simulator",
        backend_name=None,
        n_qubits=2,
        shots=100,
        device_options=None,
    )


class TestQulacsDevice:
    def test_get_device(self, qulacs_device: QulacsDevice) -> None:
        device = qulacs_device.get_device()
        assert device is not None

    def test_is_simulator(self, qulacs_device: QulacsDevice) -> None:
        assert qulacs_device.is_simulator() is True

    def test_is_remote(self, qulacs_device: QulacsDevice) -> None:
        assert qulacs_device.is_remote() is False

    def test_get_provider(self, qulacs_device: QulacsDevice) -> None:
        assert qulacs_device.get_provider() == ""

    def test_get_backend_name(self, qulacs_device: QulacsDevice) -> None:
        assert qulacs_device.get_backend_name() == ""

    def test_get_job_ids(self, qulacs_device: QulacsDevice) -> None:
        assert qulacs_device.get_job_ids() == []
