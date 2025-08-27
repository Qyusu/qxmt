from typing import cast

import numpy as np
import pennylane as qml

from qxmt.ansatze import BaseVQEAnsatz
from qxmt.devices import BaseDevice
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


class AllSinglesDoublesAnsatz(BaseVQEAnsatz):
    """All Singles and Doubles (AllSinglesDoubles) ansatz for quantum chemistry.

    The AllSinglesDoubles ansatz is a fundamental variational quantum circuit for quantum chemistry
    calculations that systematically includes all possible single and double excitations from the
    Hartree-Fock reference state. This ansatz is particularly effective for strongly correlated
    molecular systems where both single and double excitations contribute significantly to the
    ground state wavefunction.

    The ansatz constructs a quantum state by applying parameterized excitation operators:

    \|ψ⟩ = exp(∑ᵢ θᵢ Tᵢ) \|HF⟩

    where Tᵢ represents single and double excitation operators, θᵢ are variational parameters,
    and \|HF⟩ is the Hartree-Fock reference state.

    Key features:
    - Includes all chemically relevant single and double excitations
    - Maintains particle number conservation and proper spin symmetry
    - Provides systematic improvement over simpler ansätze like hardware-efficient circuits
    - Well-suited for molecules with moderate correlation effects
    - Computationally efficient compared to higher-order excitations

    The number of parameters scales polynomially with system size, making it tractable for
    near-term quantum devices while providing sufficient flexibility for accurate ground
    state preparation in many molecular systems.

    Args:
        device (BaseDevice): Quantum device for executing the variational circuit.
        hamiltonian (MolecularHamiltonian): Molecular Hamiltonian defining the quantum chemistry
            problem, including information about electrons, orbitals, and molecular geometry.

    Attributes:
        hamiltonian (MolecularHamiltonian): Molecular Hamiltonian for the ansatz.
        wires (range): Qubit indices used in the quantum circuit.
        hf_state (np.ndarray): Hartree-Fock reference state as the initial quantum state.
        singles (list): Indices of all possible single excitations from occupied to virtual orbitals.
        doubles (list): Indices of all possible double excitations from occupied to virtual orbitals.
        n_params (int): Total number of variational parameters (sum of singles and doubles).

    Example:
        >>> from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian
        >>> from qxmt.devices import BaseDevice
        >>>
        >>> # Create Hamiltonian and device for H2 molecule
        >>> hamiltonian = MolecularHamiltonian(...)
        >>> device = BaseDevice(...)
        >>>
        >>> # Initialize AllSinglesDoubles ansatz
        >>> ansatz = AllSinglesDoublesAnsatz(device, hamiltonian)
        >>>
        >>> # Initialize parameters (typically small random values)
        >>> params = np.random.normal(0, 0.01, ansatz.n_params)
        >>>
        >>> # Build and execute quantum circuit
        >>> ansatz.circuit(params)

    References:
        - PennyLane documentation: https://docs.pennylane.ai/en/stable/code/api/pennylane.AllSinglesDoubles.html

    Note:
        This ansatz is ideal for molecules where single and double excitations dominate the
        correlation energy. For systems requiring higher-order excitations, consider more
        sophisticated ansätze like kUpCCGSD or adaptive approaches.
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
            electrons=self.hamiltonian.get_active_electrons(),
            orbitals=self.hamiltonian.get_active_spin_orbitals(),
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
            electrons=self.hamiltonian.get_active_electrons(),
            orbitals=self.hamiltonian.get_active_spin_orbitals(),
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
