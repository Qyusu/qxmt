from typing import Any, Optional

from qxmt.ansatze import BaseAnsatz
from qxmt.configs import AnsatzConfig
from qxmt.devices import BaseDevice
from qxmt.hamiltonians import BaseHamiltonian
from qxmt.utils import load_object_from_yaml


class AnsatzBuilder:
    def __init__(self, config: AnsatzConfig, device: BaseDevice, hamiltonian: Optional[BaseHamiltonian] = None):
        self.config = config
        self.device = device
        self.hamiltonian = hamiltonian

    def build(self) -> BaseAnsatz:
        dynamic_params: dict[str, Any] = {"device": self.device}

        if self.hamiltonian is not None:
            dynamic_params["hamiltonian"] = self.hamiltonian

        ansatz = load_object_from_yaml(
            config=self.config.model_dump(),
            dynamic_params=dynamic_params,
        )

        if not isinstance(ansatz, BaseAnsatz):
            raise TypeError("Ansatz must be a BaseAnsatz instance.")

        return ansatz
