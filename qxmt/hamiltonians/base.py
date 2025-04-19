from abc import ABC, abstractmethod
from typing import Any


class BaseHamiltonian(ABC):
    """Base class for quantum Hamiltonians.

    This abstract base class defines the interface for Hamiltonian representations
    in quantum computing. It provides a common structure for different types of
    Hamiltonians across various quantum computing platforms.

    Args:
        platform: Name of the quantum computing platform (e.g., "pennylane").

    Attributes:
        hamiltonian: The Hamiltonian operator. Type depends on the platform.
        n_qubits: Number of qubits required to represent the Hamiltonian.
    """

    def __init__(self, platform: str) -> None:
        self.platform: str = platform
        self.hamiltonian: Any | None = None
        self.n_qubits: int | None = None

    @abstractmethod
    def get_hamiltonian(self) -> Any:
        """Get the Hamiltonian operator.

        This method must be implemented by subclasses to return the Hamiltonian
        operator in the format required by the specific platform.

        Returns:
            Any: The Hamiltonian operator. The exact type depends on the platform.

        Note:
            The returned operator should be compatible with the quantum computing
            platform specified in the constructor.
        """
        pass

    @abstractmethod
    def get_n_qubits(self) -> int:
        """Get the number of qubits required to represent the Hamiltonian.

        This method must be implemented by subclasses to return the number of
        qubits needed to represent the Hamiltonian on a quantum computer.

        Returns:
            int: Number of qubits required.

        Note:
            This value is used to validate that the quantum device has sufficient
            qubits to handle the Hamiltonian.
        """
        pass
