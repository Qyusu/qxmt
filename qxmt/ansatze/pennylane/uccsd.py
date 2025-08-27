from typing import cast

import numpy as np
import pennylane as qml

from qxmt.ansatze import BaseVQEAnsatz
from qxmt.devices import BaseDevice
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


class UCCSDAnsatz(BaseVQEAnsatz):
    """Unitary Coupled-Cluster Singles and Doubles (UCCSD) ansatz for quantum chemistry.

    The UCCSD ansatz is a cornerstone variational quantum circuit for quantum chemistry calculations
    that implements the unitary coupled-cluster approximation with singles and doubles excitations.
    This ansatz provides a quantum analog of the classical coupled-cluster theory, maintaining
    the systematic improvability and chemical accuracy while being suitable for quantum computers.

    The UCCSD ansatz constructs a quantum state through the exponential of anti-Hermitian operators:

    \|ψ⟩ = exp(T̂ - T̂†) \|HF⟩ = exp(∑ᵢ θᵢ(τᵢ - τᵢ†)) \|HF⟩

    where T̂ = T̂₁ + T̂₂ includes single (T̂₁) and double (T̂₂) excitation operators,
    θᵢ are variational parameters, and \|HF⟩ is the Hartree-Fock reference state.

    The unitary form ensures:
    - Preservation of quantum state normalization
    - Size-extensivity (energy scales correctly with system size)
    - Exact reproduction of full configuration interaction in complete basis
    - Systematic inclusion of electron correlation effects

    Key features:
    - Gold standard for quantum chemistry ansätze
    - Provides excellent accuracy for weakly to moderately correlated systems
    - Maintains proper fermion antisymmetry and particle number conservation
    - Systematic approach derived from established quantum chemistry theory
    - Serves as a benchmark for other variational ansätze

    The circuit implements Trotter approximation of the unitary exponential, making it
    practical for near-term quantum devices while preserving the essential physics
    of electron correlation.

    Args:
        device (BaseDevice): Quantum device for executing the variational circuit.
        hamiltonian (MolecularHamiltonian): Molecular Hamiltonian defining the quantum chemistry
            problem, containing electron configuration, orbital basis, and molecular geometry.

    Attributes:
        hamiltonian (MolecularHamiltonian): Molecular Hamiltonian for the ansatz.
        wires (range): Qubit indices used in the quantum circuit.
        hf_state (np.ndarray): Hartree-Fock reference state for initialization.
        singles (list): Indices of single excitations from occupied to virtual orbitals.
        doubles (list): Indices of double excitations from occupied to virtual orbitals.
        s_wires (list): Wire indices corresponding to single excitation operations.
        d_wires (list): Wire indices corresponding to double excitation operations.
        n_params (int): Total number of variational parameters.

    Example:
        >>> from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian
        >>> from qxmt.devices import BaseDevice
        >>>
        >>> # Create Hamiltonian and device for molecular system
        >>> hamiltonian = MolecularHamiltonian(...)
        >>> device = BaseDevice(...)
        >>>
        >>> # Initialize UCCSD ansatz
        >>> ansatz = UCCSDAnsatz(device, hamiltonian)
        >>>
        >>> # Initialize parameters (typically from classical coupled-cluster)
        >>> params = np.zeros(ansatz.n_params)  # or from CCSD calculation
        >>>
        >>> # Build and execute quantum circuit
        >>> ansatz.circuit(params)

    References:
        - PennyLane documentation: https://docs.pennylane.ai/en/stable/code/api/pennylane.UCCSD.html

    Note:
        UCCSD is computationally more expensive than simpler ansätze but provides superior
        accuracy for systems where single and double excitations dominate correlation effects.
        For strongly correlated systems, consider adaptive methods or higher-order excitations.

        The parameter initialization can benefit from classical coupled-cluster calculations
        to provide good starting points for the variational optimization.
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
            electrons=self.hamiltonian.get_active_electrons(),
            orbitals=self.hamiltonian.get_active_spin_orbitals(),
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
            electrons=self.hamiltonian.get_active_electrons(),
            orbitals=self.hamiltonian.get_active_spin_orbitals(),
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
