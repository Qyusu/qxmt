from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Optional

import numpy as np
import pennylane as qml
from pennylane.workflow.qnode import QNode, SupportedDiffMethods

from qxmt.ansatze import BaseAnsatz
from qxmt.devices import BaseDevice
from qxmt.hamiltonians import BaseHamiltonian
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)
DEFAULT_OPTIMIZER_STEPSIZE = 0.5


class BaseVQE(ABC):
    """Base class for Variational Quantum Eigensolver (VQE).

    This abstract base class provides the foundation for implementing VQE algorithms.
    It handles the setup of quantum circuits, parameter optimization, and tracking of
    optimization history.

    Args:
        device: Quantum device to use for the VQE calculation.
        hamiltonian: Hamiltonian to find the ground state of.
        ansatz: Quantum circuit ansatz to use.
        max_steps: Maximum number of optimization steps. Defaults to 20.
        verbose: Whether to output progress during optimization. Defaults to True.
        diff_method: Method to use for differentiation. Defaults to "adjoint".
        optimizer_settings: Settings for the optimizer.
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
        diff_method: Optional[SupportedDiffMethods] = "adjoint",
        max_steps: int = 20,
        verbose: bool = True,
        optimizer_settings: Optional[dict[str, Any]] = None,
        logger: Logger = LOGGER,
    ) -> None:
        self.device = device
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.diff_method: Optional[SupportedDiffMethods] = diff_method
        self.max_steps: int = max_steps
        self.verbose: bool = verbose
        self.optimizer_settings: Optional[dict[str, Any]] = optimizer_settings
        self.logger: Logger = logger
        self.qnode: QNode
        self.optimizer: Any
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
    def optimize(self, init_params: Any) -> None:
        """Optimize the ansatz parameters to find the ground state.

        This method must be implemented by subclasses to perform the parameter
        optimization process.

        Args:
            init_params: Initial parameters for the ansatz. This array must be calculate gradientable
                with the optimizer.

        Note:
            The implementation should update params_history and cost_history
            during the optimization process.
        """
        pass

    def _set_optimizer(self) -> None:
        """Set the optimizer based on the optimizer settings.

        This method sets the optimizer based on the optimizer settings.
        If no optimizer settings are provided, it uses the default optimizer.
        Otherwise, it uses the optimizer specified in the settings.
        """
        if self.optimizer_settings is None:
            self.optimizer = qml.GradientDescentOptimizer(stepsize=DEFAULT_OPTIMIZER_STEPSIZE)
            self.logger.info("No optimizer settings provided. Using gradient descent optimizer.")
            return
        else:
            optimizer_name = self.optimizer_settings.get("name")
            optimizer_params = self.optimizer_settings.get("params", {})

        match optimizer_name:
            case "AdagradOptimizer" | "Adagrad":
                # https://docs.pennylane.ai/en/stable/code/api/pennylane.AdagradOptimizer.html
                self.optimizer = qml.AdagradOptimizer(**optimizer_params)
            case "AdamOptimizer" | "Adam":
                # https://docs.pennylane.ai/en/stable/code/api/pennylane.AdamOptimizer.html
                self.optimizer = qml.AdamOptimizer(**optimizer_params)
            case "AdaptiveOptimizer" | "Adaptive":
                # https://docs.pennylane.ai/en/stable/code/api/pennylane.AdaptiveOptimizer.html
                self.optimizer = qml.AdaptiveOptimizer(**optimizer_params)
            case "GradientDescentOptimizer" | "GradientDescent":
                # https://docs.pennylane.ai/en/stable/code/api/pennylane.GradientDescentOptimizer.html
                self.optimizer = qml.GradientDescentOptimizer(**optimizer_params)
            case "MomentumOptimizer" | "Momentum":
                # https://docs.pennylane.ai/en/stable/code/api/pennylane.MomentumOptimizer.html
                self.optimizer = qml.MomentumOptimizer(**optimizer_params)
            case "MomentumQNGOptimizer" | "MomentumQNG":
                # https://docs.pennylane.ai/en/stable/code/api/pennylane.MomentumQNGOptimizer.html
                self.optimizer = qml.MomentumQNGOptimizer(**optimizer_params)
            case "NesterovMomentumOptimizer" | "NesterovMomentum":
                # https://docs.pennylane.ai/en/stable/code/api/pennylane.NesterovMomentumOptimizer.html
                self.optimizer = qml.NesterovMomentumOptimizer(**optimizer_params)
            case "QNGOptimizer" | "QNG":
                # https://docs.pennylane.ai/en/stable/code/api/pennylane.QNGOptimizer.html
                self.optimizer = qml.QNGOptimizer(**optimizer_params)
            case "QNSPSAOptimizer" | "QNSPSA":
                # https://docs.pennylane.ai/en/stable/code/api/pennylane.QNSPSAOptimizer.html
                self.optimizer = qml.QNSPSOptimizer(**optimizer_params)
            case "RMSPropOptimizer" | "RMSProp":
                # https://docs.pennylane.ai/en/stable/code/api/pennylane.RMSPropOptimizer.html
                self.optimizer = qml.RMSPropOptimizer(**optimizer_params)
            case "RiemannianGradientOptimizer" | "RiemannianGradient":
                # https://docs.pennylane.ai/en/stable/code/api/pennylane.RiemannianGradientOptimizer.html
                self.optimizer = qml.RiemannianGradientOptimizer(**optimizer_params)
            case "RotoselectOptimizer" | "Rotoselect":
                # https://docs.pennylane.ai/en/stable/code/api/pennylane.RotoselectOptimizer.html
                self.optimizer = qml.RotoselectOptimizer(**optimizer_params)
            case "RotosolveOptimizer" | "Rotosolve":
                # https://docs.pennylane.ai/en/stable/code/api/pennylane.RotosolveOptimizer.html
                self.optimizer = qml.RotosolveOptimizer(**optimizer_params)
            case "SPSAOptimizer" | "SPSA":
                # https://docs.pennylane.ai/en/stable/code/api/pennylane.SPSAOptimizer.html
                self.optimizer = qml.SPSAOptimizer(**optimizer_params)
            case "ShotAdaptiveOptimizer" | "ShotAdaptive":
                # https://docs.pennylane.ai/en/stable/code/api/pennylane.ShotAdaptiveOptimizer.html
                self.optimizer = qml.ShotAdaptiveOptimizer(**optimizer_params)
            case _:
                raise NotImplementedError(f'Optimizer "{optimizer_name}" is not implemented yet.')

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
