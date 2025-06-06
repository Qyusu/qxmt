from abc import ABC, abstractmethod
from enum import Enum
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
SCIPY_OPTIMIZER_PREFIX = "scipy."
INIT_PARAMS_TYPE_ZEROS = "zeros"
INIT_PARAMS_TYPE_RANDOM = "random"
INIT_PARAMS_TYPE_CUSTOM = "custom"


class OptimizerPlatform(Enum):
    SCIPY = "scipy"
    PENNYLANE = "pennylane"


class BaseVQE(ABC):
    """Base class for Variational Quantum Eigensolver (VQE).

    This abstract base class provides the foundation for implementing VQE algorithms.
    It handles the setup of quantum circuits, parameter optimization, and tracking of
    optimization history.

    Args:
        device: Quantum device to use for the VQE calculation.
        hamiltonian: Hamiltonian to find the ground state of.
        ansatz: Quantum circuit ansatz to use.
        max_steps: Maximum number of optimization steps. Defaults to 100.
        min_steps: Minimum number of optimization steps. Defaults to 1/10 of max_steps.
        tol: Tolerance for the optimization. Defaults to 1e-6.
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
        max_steps: int = 100,
        min_steps: Optional[int] = None,
        tol: float = 1e-6,
        verbose: bool = True,
        optimizer_settings: Optional[dict[str, Any]] = None,
        init_params_config: Optional[dict[str, Any]] = None,
        logger: Logger = LOGGER,
    ) -> None:
        self.device = device
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.diff_method: Optional[SupportedDiffMethods] = diff_method
        self.max_steps: int = max_steps
        self.min_steps: int = min_steps if min_steps is not None else int(max_steps * 0.1)
        self.tol: float = tol
        self.verbose: bool = verbose
        self.optimizer_settings: Optional[dict[str, Any]] = optimizer_settings
        self.init_params_config: Optional[dict[str, Any]] = init_params_config
        self.logger: Logger = logger
        self.qnode: QNode
        self.optimizer: Any
        self.params_history: list[np.ndarray] = []
        self.cost_history: list[float] = []
        self._set_optimizer_platform()
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

    def _set_optimizer_platform(self) -> None:
        """Set the optimizer platform based on the optimizer settings."""
        optimizer_name = self.optimizer_settings.get("name", "") if self.optimizer_settings else ""
        if optimizer_name.startswith(SCIPY_OPTIMIZER_PREFIX):
            self.optimizer_platform = OptimizerPlatform.SCIPY
        else:
            self.optimizer_platform = OptimizerPlatform.PENNYLANE

    def _parse_init_params(self, init_params_config: Optional[dict[str, Any]], n_params: int) -> np.ndarray:
        """Parse the initial parameters based on the init_params_config.

        Args:
            init_params_config: Configuration for the initial parameters. If None, the default is zeros.
            n_params: Number of parameters in the ansatz.

        Returns:
            Initial parameters for the ansatz.
        """
        if init_params_config is None or init_params_config.get("type") == INIT_PARAMS_TYPE_ZEROS:
            return (
                np.zeros(n_params)
                if self.optimizer_platform == OptimizerPlatform.SCIPY
                else qml.numpy.zeros(n_params, requires_grad=True)
            )
        elif init_params_config.get("type") == INIT_PARAMS_TYPE_RANDOM:
            seed = init_params_config.get("random_seed", None)
            rng = np.random.default_rng(seed)
            return (
                rng.random(n_params)
                if self.optimizer_platform == OptimizerPlatform.SCIPY
                else qml.numpy.random.rand(n_params, requires_grad=True)
            )
        elif init_params_config.get("type") == INIT_PARAMS_TYPE_CUSTOM:
            values = init_params_config.get("values", None)
            if values is None or len(values) != n_params:
                raise ValueError("Custom init_params must provide a list of length n_params")
            return (
                np.array(values)
                if self.optimizer_platform == OptimizerPlatform.SCIPY
                else qml.numpy.array(values, requires_grad=True)
            )
        else:
            raise ValueError(f"Unknown init_params type: {init_params_config.get('type')}")

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

    def _set_optimizer(self) -> None:
        """Set the optimizer based on the optimizer settings.

        This method sets the optimizer based on the optimizer settings.
        If no optimizer settings are provided, it uses the default optimizer.
        Otherwise, it uses the optimizer specified in the settings.
        """
        if self.optimizer_settings is None:
            self._set_default_optimizer()
            return

        optimizer_name = self.optimizer_settings.get("name", "")
        optimizer_params = self.optimizer_settings.get("params", {}) or {}

        if optimizer_name.startswith("scipy."):
            optimizer_params.setdefault("options", {})
            optimizer_params["options"]["maxiter"] = self.max_steps
            optimizer_params["tol"] = self.tol
            self._set_scipy_optimizer(optimizer_name, optimizer_params)
        else:
            self._set_pennylane_optimizer(optimizer_name, optimizer_params)

    def _set_default_optimizer(self) -> None:
        """Set the default gradient descent optimizer."""
        self.optimizer = qml.GradientDescentOptimizer(stepsize=DEFAULT_OPTIMIZER_STEPSIZE)
        self.logger.info("No optimizer settings provided. Using gradient descent optimizer.")

    def _set_scipy_optimizer(self, optimizer_name: str, optimizer_params: dict) -> None:
        """Set up a SciPy optimizer."""
        from scipy.optimize import minimize

        method = optimizer_name.replace("scipy.", "")

        def scipy_optimizer(params, cost_fn, **kwargs):
            return minimize(
                x0=params,
                fun=cost_fn,
                method=method,
                **optimizer_params,
                **kwargs,
            ).x

        self.optimizer = scipy_optimizer

    def _set_pennylane_optimizer(self, optimizer_name: str, optimizer_params: dict) -> None:
        """Set up a Pennylane optimizer."""
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
