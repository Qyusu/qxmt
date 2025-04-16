from abc import ABC, abstractmethod
from logging import Logger
from typing import Any

import numpy as np
import pennylane as qml
from pennylane.workflow.qnode import QNode, SupportedDiffMethods

from qxmt.ansatze import BaseAnsatz
from qxmt.devices import BaseDevice
from qxmt.hamiltonians import BaseHamiltonian
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class BaseVQE(ABC):
    """Base class for Variational Quantum Eigensolver (VQE).

    This abstract base class provides the foundation for implementing VQE algorithms.
    It handles the setup of quantum circuits, parameter optimization, and tracking of
    optimization history.

    Args:
        device: Quantum device to use for the VQE calculation.
        hamiltonian: Hamiltonian to find the ground state of.
        ansatz: Quantum circuit ansatz to use.
        diff_method: Method to use for differentiation. Defaults to "adjoint".
        logger: Logger object for output. Defaults to module-level logger.

    Attributes:
        qnode: Quantum node for executing the circuit.
        params_history: List of parameter values during optimization.
        cost_history: List of cost values during optimization.
    """

    def __init__(
        self,
        device: BaseDevice,
        hamiltonian: BaseHamiltonian,
        ansatz: BaseAnsatz,
        diff_method: SupportedDiffMethods = "adjoint",
        logger: Logger = LOGGER,
    ) -> None:
        self.device = device
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.diff_method: SupportedDiffMethods = diff_method
        self.logger: Logger = logger
        self.qnode: QNode
        self.params_history: list[np.ndarray] = []
        self.cost_history: list[float] = []
        self._initialize_qnode()

    @abstractmethod
    def _initialize_qnode(self) -> None:
        """Initialize the quantum node for VQE.

        This method must be implemented by subclasses to set up the quantum node
        that will be used for circuit execution and optimization.

        Note:
            The implementation should create a QNode that combines the ansatz circuit
            with the Hamiltonian measurement.
        """
        pass

    @abstractmethod
    def optimize(
        self,
        init_params: qml.numpy.ndarray,
        optimizer: Any,
        max_steps: int,
        verbose: bool,
    ) -> None:
        """Optimize the ansatz parameters to find the ground state.

        This method must be implemented by subclasses to perform the parameter
        optimization process.

        Args:
            init_params: Initial parameters for the ansatz.
            optimizer: Optimizer to use for parameter updates.
            max_steps: Maximum number of optimization steps.
            verbose: Whether to print progress during optimization.

        Note:
            The implementation should update params_history and cost_history
            during the optimization process.
        """
        pass

    def is_optimized(self) -> bool:
        """Check if the optimization process has been completed.

        Returns:
            bool: True if optimization has been performed (cost_history is not empty),
                 False otherwise.

        Note:
            This is a simple check that only verifies if any optimization steps
            have been taken. It does not check if the optimization has converged
            to a satisfactory solution.
        """
        return len(self.cost_history) > 0
