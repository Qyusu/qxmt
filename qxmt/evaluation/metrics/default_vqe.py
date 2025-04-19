from typing import Any, Literal, Type

from qxmt.evaluation.metrics.base import BaseMetric
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


class OptimizedEnergy(BaseMetric):
    """Metric for calculating the final optimized energy from VQE optimization history.

    This metric extracts the last energy value from the cost history, representing
    the final optimized energy obtained from the VQE algorithm.
    """

    def __init__(self, name: str = "optimized_energy") -> None:
        super().__init__(name)

    @staticmethod
    def evaluate(cost_history: list[float], **kwargs: Any) -> float:
        """Calculate the final optimized energy.

        Args:
            cost_history (list[float]): List of energy values from VQE optimization

        Returns:
            float: The final optimized energy value
        """
        return cost_history[-1]


class HF_Energy(BaseMetric):
    """Metric for calculating the Hartree-Fock (HF) energy.

    This metric computes the Hartree-Fock energy using the molecular Hamiltonian.
    """

    def __init__(self, name: str = "hf_energy") -> None:
        super().__init__(name)

    @staticmethod
    def evaluate(hamiltonian: MolecularHamiltonian, **kwargs: Any) -> float:
        """Get the precomputed Hartree-Fock energy.

        Args:
            hamiltonian (MolecularHamiltonian): The molecular Hamiltonian object

        Returns:
            float: The Hartree-Fock energy value
        """
        return hamiltonian.get_hf_energy()


class FCI_Energy(BaseMetric):
    """Metric for calculating the Full Configuration Interaction (FCI) energy.

    This metric computes the FCI energy using the molecular Hamiltonian.
    The FCI energy represents the exact solution within the given basis set.
    """

    def __init__(self, name: str = "fci_energy") -> None:
        super().__init__(name)

    @staticmethod
    def evaluate(hamiltonian: MolecularHamiltonian, **kwargs: Any) -> float | None:
        """Get the precomputed Full Configuration Interaction energy.

        Args:
            hamiltonian (MolecularHamiltonian): The molecular Hamiltonian object

        Returns:
            float | None: The FCI energy value, or None if not available
        """
        return hamiltonian.get_fci_energy()


# set default vqe metrics name list for evaluation
DEFAULT_VQE_METRICS_NAME = Literal[
    "optimized_energy",
    "hf_energy",
    "fci_energy",
]


NAME2VQE_METRIC: dict[str, Type[BaseMetric]] = {
    "optimized_energy": OptimizedEnergy,
    "hf_energy": HF_Energy,
    "fci_energy": FCI_Energy,
}
