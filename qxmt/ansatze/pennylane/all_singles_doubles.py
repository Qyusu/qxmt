from typing import cast

import numpy as np
import pennylane as qml

from qxmt.ansatze import BaseVQEAnsatz
from qxmt.devices import BaseDevice
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


class AllSinglesDoublesAnsatz(BaseVQEAnsatz):
    """AllSinglesDoubles ansatz.

    This class implements the AllSinglesDoubles ansatz for quantum chemistry calculations.
    AllSinglesDoubles is a popular ansatz for quantum chemistry that includes all possible single and double excitations
    from the Hartree-Fock reference state.

    Args:
        device (BaseDevice): Quantum device to use for the ansatz.
        hamiltonian (MolecularHamiltonian): Molecular Hamiltonian to use for the ansatz.

    Attributes:
        hamiltonian (MolecularHamiltonian): Molecular Hamiltonian to use for the ansatz.
        wires: List of qubit indices used in the circuit.
        hf_state: Hartree-Fock reference state.
        singles: List of single excitation indices.
        doubles: List of double excitation indices.
        n_params (int): Number of parameters for the AllSinglesDoubles circuit.

    URL: https://docs.pennylane.ai/en/stable/code/api/pennylane.AllSinglesDoubles.html
    """

    def __init__(self, device: BaseDevice, hamiltonian: MolecularHamiltonian) -> None:
        super().__init__(device, hamiltonian)
        self.hamiltonian = cast(MolecularHamiltonian, self.hamiltonian)
        self.wires = range(self.hamiltonian.get_n_qubits())
        self.prepare_hf_state()
        self.prepare_excitation()
        self.n_params = len(self.singles) + len(self.doubles)

    def prepare_hf_state(self) -> None:
        """Prepare the Hartree-Fock reference state.

        This method creates the Hartree-Fock reference state using PennyLane's qchem module.
        The Hartree-Fock state is a product state where the first n electrons occupy the lowest
        energy orbitals.

        The state is stored in self.hf_state as a numpy array.

        Note:
            The number of electrons and orbitals are obtained from the Hamiltonian.
        """
        self.hf_state = qml.qchem.hf_state(
            electrons=self.hamiltonian.get_active_electrons(), orbitals=self.hamiltonian.get_active_spin_orbitals()
        )

    def prepare_excitation(self) -> None:
        """Prepare the single and double excitations.

        This method generates all possible single and double excitations from the Hartree-Fock state.

        The results are stored in the following attributes:
        - self.singles: List of single excitation indices
        - self.doubles: List of double excitation indices

        Note:
            The number of excitations depends on the number of electrons and orbitals in the system.
        """
        self.singles, self.doubles = qml.qchem.excitations(
            electrons=self.hamiltonian.get_active_electrons(), orbitals=self.hamiltonian.get_active_spin_orbitals()
        )

    def circuit(self, params: np.ndarray) -> None:
        """Construct the AllSinglesDoubles quantum circuit.

        Args:
            params (np.ndarray): Parameters for the AllSinglesDoubles circuit. The length of this array
                               should match the number of single and double excitations.

        Note:
            The AllSinglesDoubles operation includes all possible single and double excitations from the
            Hartree-Fock reference state.
        """
        qml.AllSinglesDoubles(
            weights=params,
            wires=self.wires,
            hf_state=self.hf_state,
            singles=self.singles,
            doubles=self.doubles,
        )
