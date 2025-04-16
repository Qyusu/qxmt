from qxmt.ansatze import AnsatzBuilder, BaseAnsatz
from qxmt.configs import ExperimentConfig
from qxmt.constants import DEFAULT_N_JOBS
from qxmt.devices import BaseDevice, DeviceBuilder
from qxmt.exceptions import InvalidModelNameError
from qxmt.hamiltonians import BaseHamiltonian, HamiltonianBuilder
from qxmt.models.vqe.base import BaseVQE
from qxmt.models.vqe.basic import BasicVQE


class VQEBuilder:
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
                self.model = BasicVQE(self.device, self.hamiltonian, self.ansatz)
            case _:
                raise InvalidModelNameError(f'"{self.config.model.name}" is not implemented.')

    def build(self) -> BaseVQE:
        """Build the VQE model."""
        self._set_device()
        self._set_hamiltonian()
        self._set_ansatz()
        self._set_model()

        return self.model
