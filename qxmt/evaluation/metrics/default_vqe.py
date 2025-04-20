from typing import Any, Literal, Type

from qxmt.evaluation.metrics.base import BaseMetric
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


class FinalCost(BaseMetric):
    """Metric for calculating the final cost from VQE optimization history.

    This metric extracts the last cost value from the cost history, representing
    the final cost obtained from the VQE algorithm.
    """

    def __init__(self, name: str = "final_cost") -> None:
        super().__init__(name)

    @staticmethod
    def evaluate(cost_history: list[float], **kwargs: Any) -> float:
        """Calculate the final cost.

        Args:
            cost_history (list[float]): List of cost values from VQE optimization

        Returns:
            float: The final cost value
        """
        return cost_history[-1]


class HFEnergy(BaseMetric):
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


class FCIEnergy(BaseMetric):
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
    "final_cost",
    "hf_energy",
    "fci_energy",
]


NAME2VQE_METRIC: dict[str, Type[BaseMetric]] = {
    "final_cost": FinalCost,
    "hf_energy": HFEnergy,
    "fci_energy": FCIEnergy,
}
