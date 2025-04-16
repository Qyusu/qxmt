from qxmt.configs import HamiltonianConfig
from qxmt.hamiltonians import BaseHamiltonian
from qxmt.utils import load_object_from_yaml


class HamiltonianBuilder:
    def __init__(self, config: HamiltonianConfig):
        self.config = config

    def build(self) -> BaseHamiltonian:
        self.hamiltonian = load_object_from_yaml(config=self.config.model_dump())

        if not isinstance(self.hamiltonian, BaseHamiltonian):
            raise TypeError("Hamiltonian must be a BaseHamiltonian instance.")

        return self.hamiltonian
