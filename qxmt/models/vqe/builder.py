from typing import cast

from pennylane.workflow.qnode import SupportedDiffMethods

from qxmt.ansatze import AnsatzBuilder, BaseAnsatz
from qxmt.configs import ExperimentConfig
from qxmt.constants import DEFAULT_N_JOBS
from qxmt.devices import BaseDevice, DeviceBuilder
from qxmt.exceptions import InvalidModelNameError
from qxmt.hamiltonians import BaseHamiltonian, HamiltonianBuilder
from qxmt.models.vqe.base import BaseVQE
from qxmt.models.vqe.basic import BasicVQE


class VQEModelBuilder:
    """Builder class for VQE models."""

    def __init__(self, config: ExperimentConfig, n_jobs: int = DEFAULT_N_JOBS) -> None:
        """Initialize the model builder."""
        self.config: ExperimentConfig = config
        self.n_jobs: int = n_jobs
        self.device: BaseDevice
        self.hamiltonian: BaseHamiltonian
        self.ansatz: BaseAnsatz
        self.model: BaseVQE

    def _set_device(self) -> None:
        device_config = self.config.device
        self.device = DeviceBuilder(config=device_config).build()

    def _set_hamiltonian(self) -> None:
        hamiltonian_config = self.config.hamiltonian
        if hamiltonian_config is None:
            raise ValueError("Hamiltonian config is not provided.")

        self.hamiltonian = HamiltonianBuilder(config=hamiltonian_config).build()

    def _set_ansatz(self) -> None:
        ansatz_config = self.config.ansatz
        if ansatz_config is None:
            raise ValueError("Ansatz config is not provided.")

        self.ansatz = AnsatzBuilder(config=ansatz_config, device=self.device, hamiltonian=self.hamiltonian).build()

    def _set_model(self) -> None:
        match self.config.model.name:
            case "basic":
                self.model = BasicVQE(
                    device=self.device,
                    hamiltonian=self.hamiltonian,
                    ansatz=self.ansatz,
                    diff_method=cast(SupportedDiffMethods, self.config.model.diff_method),
                    max_steps=self.config.model.params.get("max_steps", 20),
                    min_steps=self.config.model.params.get("min_steps", None),
                    tol=self.config.model.params.get("tol", 1e-6),
                    verbose=self.config.model.params.get("verbose", True),
                    optimizer_settings=self.config.model.optimizer_settings,
                    init_params_config=self.config.model.params.get("init_params", None),
                )
            case _:
                raise InvalidModelNameError(f'"{self.config.model.name}" is not implemented.')

    def build(self) -> BaseVQE:
        """Build the VQE model."""
        self._set_device()
        self._set_hamiltonian()
        self._set_ansatz()
        self._set_model()

        return self.model
