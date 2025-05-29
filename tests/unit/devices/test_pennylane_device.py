import pytest

from qxmt.devices.pennylane_device import PennyLaneDevice


@pytest.fixture
def simulator_device() -> PennyLaneDevice:
    return PennyLaneDevice(
        platform="pennylane",
        device_name="default.qubit",
        backend_name=None,
        n_qubits=2,
        shots=100,
    )


class TestPennyLaneDevice:
    def test_get_device(self, simulator_device: PennyLaneDevice):
        device = simulator_device.get_device()
        assert device is not None

    def test_is_simulator(self, simulator_device: PennyLaneDevice):
        assert simulator_device.is_simulator() is True

    def test_is_remote(self, simulator_device: PennyLaneDevice):
        assert simulator_device.is_remote() is False

    def test_get_provider(self, simulator_device: PennyLaneDevice):
        assert simulator_device.get_provider() == ""

    def test_get_job_ids(self, simulator_device: PennyLaneDevice):
        assert simulator_device.get_job_ids() == []
