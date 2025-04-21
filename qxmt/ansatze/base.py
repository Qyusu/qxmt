from abc import ABC, abstractmethod
from logging import Logger
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from qxmt.devices import BaseDevice
from qxmt.hamiltonians import BaseHamiltonian
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class BaseAnsatz(ABC):
    """Base class for quantum circuit ansatzes.

    This abstract base class defines the interface for quantum circuit ansatzes.
    It provides common functionality for circuit visualization and execution.

    Args:
        device: Quantum device to use for the ansatz.

    Note:
        Subclasses must implement the circuit method to define the quantum circuit.
    """

    def __init__(self, device: BaseDevice) -> None:
        self.device = device
        self.n_params: int

    @abstractmethod
    def circuit(self, *args, **kwargs) -> None:
        """Define the quantum circuit for the ansatz.

        This method must be implemented by subclasses to define the quantum circuit
        structure and operations.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Note:
            The implementation should define the quantum circuit using the device's
            operations.
        """
        pass

    def draw(
        self,
        params: np.ndarray,
        format: str = "default",
        logger: Logger = LOGGER,
        **kwargs: Any,
    ) -> None:
        """Draw the quantum circuit using the platform's draw function.

        This method visualizes the quantum circuit in either text or graphical format.

        Args:
            params: Parameters for the circuit.
            format: Format of the drawing. Choose between "default" (text) or "mpl" (matplotlib).
                   Defaults to "default".
            logger: Logger object for output. Defaults to module-level logger.
            **kwargs: Additional arguments for the drawing function.

        Raises:
            NotImplementedError: If the platform is not supported.
            ValueError: If the format is invalid.

        Note:
            For matplotlib format, the figure will be displayed using plt.show().
        """
        platform = self.device.platform
        if platform == "pennylane":
            import pennylane as qml

            qnode = qml.QNode(self.circuit, self.device.get_device())
            match format:
                case "default":
                    drawer = qml.draw(qnode, **kwargs)
                    logger.info(drawer(params))
                case "mpl":
                    fig, _ = qml.draw_mpl(qnode, **kwargs)(params)
                    plt.show()
                case _:
                    raise ValueError(f"Invalid format '{format}' for drawing the circuit")
        else:
            raise NotImplementedError(f'"draw" method is not supported in {platform}.')


class BaseVQEAnsatz(BaseAnsatz):
    """Base class for Variational Quantum Eigensolver (VQE) ansatzes.

    This abstract base class extends BaseAnsatz to provide functionality specific
    to VQE calculations, including Hamiltonian validation and expectation value
    measurement.

    Args:
        device: Quantum device to use for the ansatz.
        hamiltonian: Hamiltonian to use for the VQE calculation.

    Note:
        Subclasses must implement the circuit method to define the variational circuit.
    """

    def __init__(self, device: BaseDevice, hamiltonian: BaseHamiltonian) -> None:
        super().__init__(device)
        self.hamiltonian = hamiltonian
        self._validate_qubit_count()

    def _validate_qubit_count(self) -> None:
        """Validate that the Hamiltonian's qubit count is compatible with the device.

        This method checks that:
        1. The Hamiltonian has a defined qubit count
        2. The Hamiltonian's qubit count does not exceed the device's capacity

        Raises:
            ValueError: If the Hamiltonian's qubit count is not set or exceeds the device's capacity.
        """
        if self.hamiltonian.n_qubits is None:
            raise ValueError("The qubit count of the Hamiltonian is not set.")

        if self.hamiltonian.n_qubits > self.device.n_qubits:
            raise ValueError(
                f"The qubit count of the Hamiltonian ({self.hamiltonian.n_qubits}) "
                f"exceeds the device ({self.device.n_qubits}). "
                f"Please check the qubit count of the Hamiltonian and the device."
            )

    @abstractmethod
    def circuit(self, params: np.ndarray) -> None:
        """Define the variational circuit for the VQE ansatz.

        This method must be implemented by subclasses to define the variational circuit.

        Args:
            params: Parameters for the variational circuit.

        Note:
            The implementation should define a parameterized quantum circuit that
            can be optimized to find the ground state of the Hamiltonian.
        """
        pass
