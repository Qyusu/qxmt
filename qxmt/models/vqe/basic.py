from logging import Logger
from typing import Any, Optional, cast

import pennylane as qml
from pennylane.measurements import ExpectationMP
from pennylane.ops.op_math import Sum
from pennylane.workflow.qnode import SupportedDiffMethods

from qxmt.ansatze import BaseAnsatz
from qxmt.devices import BaseDevice
from qxmt.hamiltonians import BaseHamiltonian
from qxmt.logger import set_default_logger
from qxmt.models.vqe.base import BaseVQE

LOGGER = set_default_logger(__name__)

DEFAULT_STEPSIZE = 0.5


class BasicVQE(BaseVQE):
    """Basic implementation of the Variational Quantum Eigensolver (VQE).

    This class provides a basic implementation of VQE using PennyLane's optimization tools.
    It supports gradient-based optimization of quantum circuits to find the ground state
    energy of a given Hamiltonian.

    Args:
        device: Quantum device to use for the VQE.
        hamiltonian: Hamiltonian to find the ground state of.
        ansatz: Quantum circuit ansatz to use.
        diff_method: Method to use for differentiation. Defaults to "adjoint".
        optimizer_settings: Settings for the optimizer.
        logger: Logger object for output. Defaults to module-level logger.

    Attributes:
        cost_history: List of cost values during optimization.
        params_history: List of parameter values during optimization.
    """

    def __init__(
        self,
        device: BaseDevice,
        hamiltonian: BaseHamiltonian,
        ansatz: BaseAnsatz,
        diff_method: Optional[SupportedDiffMethods] = "adjoint",
        optimizer_settings: Optional[dict[str, Any]] = None,
        logger: Logger = LOGGER,
    ) -> None:
        super().__init__(device, hamiltonian, ansatz, diff_method, optimizer_settings, logger)

    def _initialize_qnode(self) -> None:
        """Initialize the QNode for VQE.

        This method creates a quantum node that:
        1. Executes the ansatz circuit with given parameters
        2. Measures the expectation value of the Hamiltonian

        Raises:
            ValueError: If the Hamiltonian is not a Sum instance.
        """

        def circuit_with_measurement(params: qml.numpy.ndarray) -> ExpectationMP:
            self.ansatz.circuit(params)
            if not isinstance(self.hamiltonian.hamiltonian, Sum):
                raise ValueError("Hamiltonian must be a Sum instance.")
            else:
                return qml.expval(self.hamiltonian.hamiltonian)

        self.qnode = qml.QNode(
            func=circuit_with_measurement,
            device=self.device.get_device(),
            diff_method=cast(SupportedDiffMethods, self.diff_method),
        )

    def optimize(
        self,
        init_params: qml.numpy.ndarray,
        max_steps: int = 100,
        verbose: bool = True,
    ) -> None:
        """Optimize the ansatz parameters to find the ground state.

        This method performs gradient-based optimization of the ansatz parameters
        to minimize the expectation value of the Hamiltonian.

        Args:
            init_params: Initial parameters for the ansatz.
            max_steps: Maximum number of optimization steps. Defaults to 100.
            verbose: Whether to output progress during optimization. Defaults to True.

        Note:
            The optimization history (cost and parameters) is stored in the class attributes
            cost_history and params_history.
        """
        self._set_optimizer()
        self.logger.info(f"Optimizing ansatz with {len(init_params)} parameters through {max_steps} steps")
        params = init_params
        for i in range(max_steps):
            params, cost = self.optimizer.step_and_cost(self.qnode, params)
            self.cost_history.append(cost)
            self.params_history.append(params)
            if verbose:
                self.logger.info(f"Step {i+1}: Cost = {cost}")

        self.logger.info(f"Optimization finished. Final cost: {self.cost_history[-1]:.8f}")
