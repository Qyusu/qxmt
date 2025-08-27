import math
from typing import cast

import numpy as np
import pennylane as qml

from qxmt.ansatze import BaseVQEAnsatz
from qxmt.devices import BaseDevice
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


class KUpCCGSDAnsatz(BaseVQEAnsatz):
    """k-Unitary Pair Coupled Cluster with Generalized Singles and Doubles (kUpCCGSD) ansatz.

    The kUpCCGSD ansatz is an extension of the Unitary Pair Coupled Cluster (UpCC) framework that
    incorporates both generalized single and double excitations. This ansatz is particularly
    suitable for quantum chemistry calculations as it maintains physical properties such as
    particle number conservation and spin symmetry while providing a variational quantum circuit
    for molecular ground state preparation.

    The ansatz constructs a quantum state by applying k repetitions of the UpCCGSD unitary operator
    to the Hartree-Fock reference state:

    \|ψ⟩ = [U_CCGSD(θ)]^k \|HF⟩

    where U_CCGSD(θ) includes both single and double excitation operators with parameters θ.
    The generalized singles include additional spin-flip excitations controlled by the delta_sz
    parameter, allowing for more flexible electronic structure descriptions.

    Key features:
    - Maintains particle number and spin conservation
    - Includes generalized single excitations with spin selection rules
    - Supports multiple repetitions (k) of the unitary operator for enhanced expressibility
    - Compatible with various molecular systems and basis sets

    Args:
        device (BaseDevice): Quantum device to use for executing the ansatz circuit.
        hamiltonian (MolecularHamiltonian): Molecular Hamiltonian defining the quantum chemistry
            problem. Must contain information about electrons, orbitals, and molecular geometry.
        k (int, optional): Number of repetitions of the UpCCGSD unitary operator. Higher values
            increase circuit depth but may improve state preparation accuracy. Defaults to 1.
        delta_sz (int, optional): Spin selection rule for generalized single excitations.
            Specifies the allowed change in spin projection (sz[p] - sz[r] = delta_sz) where
            p and r are orbital indices. Valid values are 0, +1, and -1. Defaults to 0.

    Attributes:
        hamiltonian (MolecularHamiltonian): Molecular Hamiltonian for the ansatz.
        wires (range): Qubit indices used in the quantum circuit.
        k (int): Number of UpCCGSD unitary repetitions.
        delta_sz (int): Spin selection rule parameter for generalized singles.
        hf_state (np.ndarray): Hartree-Fock reference state as initial quantum state.
        params_shape (tuple): Shape of the parameter array required for the ansatz.
        n_params (int): Total number of variational parameters.

    Example:
        >>> from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian
        >>> from qxmt.devices import BaseDevice
        >>>
        >>> # Create Hamiltonian and device
        >>> hamiltonian = MolecularHamiltonian(...)
        >>> device = BaseDevice(...)
        >>>
        >>> # Initialize kUpCCGSD ansatz with 2 repetitions
        >>> ansatz = KUpCCGSDAnsatz(device, hamiltonian, k=2, delta_sz=0)
        >>>
        >>> # Get parameter shape and initialize parameters
        >>> param_shape = ansatz.params_shape
        >>> params = np.random.normal(0, 0.1, ansatz.n_params)
        >>>
        >>> # Build quantum circuit
        >>> ansatz.circuit(params)

    References:
        - PennyLane documentation: https://docs.pennylane.ai/en/stable/code/api/pennylane.kUpCCGSD.html

    Note:
        This ansatz is designed for near-term quantum devices and provides a balance between
        circuit depth and expressibility. The choice of k and delta_sz parameters should be
        optimized based on the specific molecular system and available quantum hardware.
    """

    def __init__(
        self,
        device: BaseDevice,
        hamiltonian: MolecularHamiltonian,
        k: int = 1,
        delta_sz: int = 0,
    ) -> None:
        super().__init__(device, hamiltonian)
        self.hamiltonian = cast(MolecularHamiltonian, self.hamiltonian)
        self.wires = range(self.hamiltonian.get_n_qubits())
        self.k = k
        self.delta_sz = delta_sz
        self.prepare_hf_state()
        self.params_shape = qml.kUpCCGSD.shape(n_wires=len(self.wires), k=self.k, delta_sz=self.delta_sz)
        self.n_params = math.prod(self.params_shape)

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
            params (np.ndarray): Parameters for the KUpCCGSD circuit. The length of this array
                               should match the number of parameters required by the circuit.
        """
        qml.kUpCCGSD(
            weights=params.reshape(self.params_shape),
            wires=self.wires,
            k=self.k,
            delta_sz=self.delta_sz,
            init_state=self.hf_state,
        )
