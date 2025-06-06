from typing import Any, Callable

import pennylane as qml
import pytest

from qxmt.ansatze import BaseAnsatz
from qxmt.devices.pennylane_device import PennyLaneDevice
from qxmt.hamiltonians import BaseHamiltonian
from qxmt.models.vqe.base import BaseVQE

DEVICE = PennyLaneDevice(platform="pennylane", device_name="default.qubit", backend_name=None, n_qubits=3, shots=None)


class DummyHamiltonian(BaseHamiltonian):
    def __init__(self):
        super().__init__(platform="pennylane")
        self.n_qubits = 3

    def get_hamiltonian(self):
        return None

    def get_n_qubits(self) -> int:
        return 3


class DummyAnsatz(BaseAnsatz):
    def __init__(self):
        super().__init__(device=DEVICE)
        self.n_params = 3

    def circuit(self, *args, **kwargs):
        pass


class DummyVQE(BaseVQE):
    def _initialize_qnode(self):
        self.qnode = qml.QNode(lambda x: x, DEVICE.get_device())

    def optimize(self, init_params):
        self.params_history.append(init_params)
        self.cost_history.append(0.0)


@pytest.fixture(scope="function")
def build_vqe() -> Callable:
    def _build_vqe(**kwargs: Any) -> BaseVQE:
        return DummyVQE(device=DEVICE, hamiltonian=DummyHamiltonian(), ansatz=DummyAnsatz(), **kwargs)

    return _build_vqe
