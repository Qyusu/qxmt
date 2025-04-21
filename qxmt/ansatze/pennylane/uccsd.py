from typing import cast

import numpy as np
import pennylane as qml

from qxmt.ansatze import BaseVQEAnsatz
from qxmt.devices import BaseDevice
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


class UCCSDAnsatz(BaseVQEAnsatz):
    """Unitary Coupled-Cluster Singles and Doubles (UCCSD) ansatz.

    This class implements the UCCSD ansatz for quantum chemistry calculations.
    UCCSD is a popular ansatz for quantum chemistry that includes single and double excitations
    from the Hartree-Fock reference state.

    Args:
        device (BaseDevice): Quantum device to use for the ansatz.
        hamiltonian (MolecularHamiltonian): Molecular Hamiltonian to use for the ansatz.

    Attributes:
        wires: List of qubit indices used in the circuit.
        hf_state: Hartree-Fock reference state.
        singles: List of single excitation indices.
        doubles: List of double excitation indices.
        s_wires: List of wire indices for single excitations.
        d_wires: List of wire indices for double excitations.

    Example:
        >>> from qxmt.devices.pennylane import DefaultQubit
        >>> from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian
        >>> device = DefaultQubit(n_qubits=4)
        >>> hamiltonian = MolecularHamiltonian(...)
        >>> ansatz = UCCSDAnsatz(device, hamiltonian)
        >>> params = np.random.rand(10)  # Number of parameters depends on the system
        >>> ansatz.circuit(params)

    Note:
        The number of parameters required depends on the number of single and double excitations,
        which is determined by the number of electrons and orbitals in the system.

    URL: https://docs.pennylane.ai/en/stable/code/api/pennylane.UCCSD.html
    """

    def __init__(self, device: BaseDevice, hamiltonian: MolecularHamiltonian) -> None:
        super().__init__(device, hamiltonian)
        self.hamiltonian = cast(MolecularHamiltonian, self.hamiltonian)
        self.wires = range(self.hamiltonian.get_n_qubits())
        self.prepare_hf_state()
        self.prepare_excitation_wires()
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

    def prepare_excitation_wires(self) -> None:
        """Prepare the excitation wires for single and double excitations.

        This method:
        1. Generates all possible single and double excitations from the Hartree-Fock state
        2. Converts these excitations to wire indices that can be used in the quantum circuit

        The results are stored in the following attributes:
        - self.singles: List of single excitation indices
        - self.doubles: List of double excitation indices
        - self.s_wires: List of wire indices for single excitations
        - self.d_wires: List of wire indices for double excitations

        Note:
            The number of excitations depends on the number of electrons and orbitals in the system.
        """
        self.singles, self.doubles = qml.qchem.excitations(
            electrons=self.hamiltonian.get_active_electrons(), orbitals=self.hamiltonian.get_active_spin_orbitals()
        )
        self.s_wires, self.d_wires = qml.qchem.excitations_to_wires(self.singles, self.doubles)

    def circuit(self, params: np.ndarray) -> None:
        """Construct the UCCSD quantum circuit.

        Args:
            params (np.ndarray): Parameters for the UCCSD circuit. The length of this array
                               should match the number of single and double excitations.

        Note:
            The UCCSD operation includes both single and double excitations from the
            Hartree-Fock reference state.
        """
        qml.UCCSD(params, self.wires, self.s_wires, self.d_wires, self.hf_state)
