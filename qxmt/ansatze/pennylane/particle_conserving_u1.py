import math
from typing import cast

import numpy as np
import pennylane as qml

from qxmt.ansatze import BaseVQEAnsatz
from qxmt.devices import BaseDevice
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


class ParticleConservingU1Ansatz(BaseVQEAnsatz):
    """Particle-Conserving U(1) ansatz for quantum chemistry calculations.

    The ParticleConservingU1 ansatz is a hardware-efficient variational quantum circuit that
    respects U(1) particle number symmetry, making it particularly well-suited for quantum
    chemistry applications where particle number conservation is essential. This ansatz
    provides an excellent balance between expressibility and physical constraints.

    The circuit architecture consists of layers of parameterized gates that preserve the
    total particle number throughout the computation. Each layer contains:
    1. Single-qubit rotations that respect particle number conservation
    2. Two-qubit gates that maintain the U(1) symmetry
    3. Controlled operations that preserve fermion number

    The ansatz constructs quantum states of the form:

    \|ψ⟩ = U_U1^(L)(θ_L) ⋯ U_U1^(1)(θ_1) \|init⟩

    where U_U1^(i) represents the i-th particle-conserving layer, θ_i are variational
    parameters, and \|init⟩ is the initial state (typically Hartree-Fock).

    Key features:
    - Strict particle number conservation (exact U(1) symmetry)
    - Hardware-efficient design for NISQ devices
    - Reduced parameter space due to symmetry constraints
    - Natural incorporation of physical quantum chemistry constraints
    - Improved optimization landscape compared to unconstrained ansätze
    - Compatible with fermionic mappings (Jordan-Wigner, Bravyi-Kitaev)

    The U(1) symmetry constraint significantly reduces the parameter space and helps
    avoid unphysical states during optimization, leading to more stable and efficient
    variational quantum eigensolvers.

    Args:
        device (BaseDevice): Quantum device for executing the variational circuit.
        hamiltonian (MolecularHamiltonian): Molecular Hamiltonian defining the quantum chemistry
            problem, containing electron configuration, orbital basis, and molecular geometry.
        n_layers (int, optional): Number of circuit layers. Each layer applies particle-conserving
            operations while maintaining U(1) symmetry. Higher values increase expressibility
            but also circuit depth. Defaults to 2.

    Attributes:
        hamiltonian (MolecularHamiltonian): Molecular Hamiltonian for the ansatz.
        n_layers (int): Number of layers in the circuit architecture.
        wires (range): Qubit indices used in the quantum circuit.
        hf_state (np.ndarray): Hartree-Fock reference state for initialization.
        params_shape (tuple): Shape of the parameter tensor for the circuit.
        n_params (int): Total number of variational parameters.

    Example:
        >>> from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian
        >>> from qxmt.devices import BaseDevice
        >>>
        >>> # Create Hamiltonian and device
        >>> hamiltonian = MolecularHamiltonian(...)
        >>> device = BaseDevice(...)
        >>>
        >>> # Initialize ParticleConservingU1 ansatz with 4 layers
        >>> ansatz = ParticleConservingU1Ansatz(device, hamiltonian, n_layers=4)
        >>>
        >>> # Initialize parameters
        >>> params = np.random.normal(0, 0.02, ansatz.n_params)
        >>>
        >>> # Build quantum circuit (particle number automatically conserved)
        >>> ansatz.circuit(params)

    References:

        - PennyLane documentation: https://docs.pennylane.ai/en/stable/code/api/pennylane.ParticleConservingU1.html

    Note:
        This ansatz is particularly beneficial for systems where particle number conservation
        is crucial for physical accuracy. The U(1) symmetry constraint reduces the search
        space and can lead to faster convergence compared to unconstrained variational circuits.

        The reduced parameter space makes this ansatz especially suitable for near-term
        quantum devices with limited coherence times, as it requires fewer optimization steps
        while maintaining chemical accuracy.
    """

    def __init__(self, device: BaseDevice, hamiltonian: MolecularHamiltonian, n_layers: int = 2) -> None:
        super().__init__(device, hamiltonian)
        self.hamiltonian = cast(MolecularHamiltonian, self.hamiltonian)
        self.n_layers = n_layers
        self.wires = range(self.hamiltonian.get_n_qubits())
        self.prepare_hf_state()
        self.params_shape = qml.ParticleConservingU1.shape(n_wires=len(self.wires), n_layers=self.n_layers)
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

    def circuit(self, params: np.ndarray) -> None:
        """Construct the ParticleConservingU1 quantum circuit.

        This method builds a quantum circuit that preserves particle number through U(1) symmetry.
        The circuit applies multiple layers of particle-conserving gates while maintaining the
        total number of particles (electrons) in the system.

        Args:
            params (np.ndarray): Parameters for the ParticleConservingU1 circuit. The length of this
                               array should match the number of parameters required by the circuit
                               (determined by n_layers and number of qubits).
        """
        qml.ParticleConservingU1(
            weights=params.reshape(self.params_shape),
            wires=self.wires,
            init_state=self.hf_state,
        )
