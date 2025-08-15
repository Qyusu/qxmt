from logging import Logger
from typing import Any, Optional, cast

import numpy as np
import pennylane as qml
from pennylane.measurements import ExpectationMP
from pennylane.ops.op_math import Sum
from pennylane.workflow.qnode import SupportedDiffMethods

from qxmt.ansatze import BaseAnsatz
from qxmt.devices import BaseDevice
from qxmt.hamiltonians import BaseHamiltonian
from qxmt.logger import set_default_logger
from qxmt.models.vqe.base import BaseVQE, OptimizerPlatform

LOGGER = set_default_logger(__name__)


class BasicVQE(BaseVQE):
    """Basic implementation of the Variational Quantum Eigensolver (VQE).

    This class provides a basic implementation of VQE using PennyLane's optimization tools.
    It supports gradient-based optimization of quantum circuits to find the ground state
    energy of a given Hamiltonian. SciPy optimizers can now use PennyLane's automatic
    differentiation for efficient gradient computation.

    Args:
        device: Quantum device to use for the VQE.
        hamiltonian: Hamiltonian to find the ground state of.
        ansatz: Quantum circuit ansatz to use.
        diff_method: Method to use for differentiation. Defaults to "adjoint".
        max_steps: Maximum number of optimization steps. Defaults to 100.
        min_steps: Minimum number of optimization steps. Defaults to 1/10 of max_steps.
        tol: Tolerance for the optimization. Defaults to 1e-6.
        verbose: Whether to output progress during optimization. Defaults to True.
        optimizer_settings: Settings for the optimizer. For SciPy optimizers, you can
            set "gradient_method": "numerical" to use SciPy's numerical differentiation
            instead of PennyLane's automatic differentiation.
        logger: Logger object for output. Defaults to module-level logger.

    Example:
        Using SciPy optimizer with PennyLane automatic differentiation::

            optimizer_settings = {
                "name": "scipy.BFGS",
                "params": {
                    "gradient_method": "autodiff",  # Use PennyLane autodiff (default)
                    "options": {"gtol": 1e-6}
                }
            }
            vqe = BasicVQE(device, hamiltonian, ansatz, optimizer_settings=optimizer_settings)
            vqe.optimize()

        Using SciPy numerical differentiation::

            optimizer_settings = {
                "name": "scipy.BFGS",
                "params": {
                    "gradient_method": "numerical",  # Use SciPy numerical gradients
                }
            }

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
        max_steps: int = 100,
        min_steps: Optional[int] = None,
        tol: float = 1e-6,
        verbose: bool = True,
        optimizer_settings: Optional[dict[str, Any]] = None,
        init_params_config: Optional[dict[str, Any]] = None,
        logger: Logger = LOGGER,
    ) -> None:
        super().__init__(
            device=device,
            hamiltonian=hamiltonian,
            ansatz=ansatz,
            diff_method=diff_method,
            max_steps=max_steps,
            min_steps=min_steps,
            tol=tol,
            verbose=verbose,
            optimizer_settings=optimizer_settings,
            init_params_config=init_params_config,
            logger=logger,
        )

    def circuit_with_measurement(self, params: qml.numpy.ndarray | np.ndarray) -> ExpectationMP:
        self.ansatz.circuit(params)
        if not isinstance(self.hamiltonian.hamiltonian, Sum):
            raise ValueError("Hamiltonian must be a Sum instance.")
        else:
            return qml.expval(self.hamiltonian.hamiltonian)

    def _initialize_qnode(self) -> None:
        """Initialize the QNode for VQE.

        This method creates a quantum node that:
        1. Executes the ansatz circuit with given parameters
        2. Measures the expectation value of the Hamiltonian

        Raises:
            ValueError: If the Hamiltonian is not a Sum instance.
        """
        self.qnode = qml.QNode(
            func=self.circuit_with_measurement,
            device=self.device.get_device(),
            diff_method=cast(SupportedDiffMethods, self.diff_method),
        )

    def _optimize_scipy(self, init_params: np.ndarray) -> None:
        """Optimize the ansatz parameters using scipy.

        Args:
            init_params (np.ndarray): Initial parameters for the ansatz.
        """
        step_num = {"step": 0}

        def cost_function(params):
            cost = self.qnode(params)
            return float(cost)

        def callback(params):
            cost = self.qnode(params)
            self.cost_history.append(float(cost))
            self.params_history.append(params.copy())
            step_num["step"] += 1
            if self.verbose:
                self.logger.info(f"Step {step_num['step']}: Cost = {cost}")

        self.optimizer(
            init_params,
            cost_function,
            callback=callback,
        )

    def _optimize_pennylane(self, init_params: qml.numpy.ndarray) -> None:
        """Optimize the ansatz parameters using Pennylane.

        Args:
            init_params (qml.numpy.ndarray): Initial parameters for the ansatz.
        """
        params = init_params
        for i in range(self.max_steps):
            params, cost = self.optimizer.step_and_cost(self.qnode, params)
            self.cost_history.append(cost)
            self.params_history.append(params)
            if self.verbose:
                self.logger.info(f"Step {i+1}: Cost = {cost}")

            if i > self.min_steps and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tol:
                self.logger.info(f"Optimization finished at step {i+1}.")
                break

    def optimize(self, init_params: Optional[qml.numpy.ndarray | np.ndarray] = None) -> None:
        """Optimize the ansatz parameters to find the ground state.

        This method performs gradient-based optimization of the ansatz parameters
        to minimize the expectation value of the Hamiltonian.

        Args:
            init_params (Optional[qml.numpy.ndarray | np.ndarray]): Initial parameters for the ansatz.
                If None, the ansatz parameters are initialized to zero.

        Note:
            The optimization history (cost and parameters) is stored in the class attributes
            cost_history and params_history.
        """
        self._set_optimizer()
        self.logger.info(f"Optimizing ansatz with {self.ansatz.n_params} parameters through {self.max_steps} steps")

        if init_params is None:
            init_params = self._parse_init_params(self.init_params_config, self.ansatz.n_params)

        self._set_circuit_depth(self.qnode, init_params)

        if self.optimizer_platform == OptimizerPlatform.SCIPY:
            self._optimize_scipy(init_params)
        else:
            self._optimize_pennylane(init_params)

        self.logger.info(f"Optimization finished. Final cost: {self.cost_history[-1]:.8f}")
