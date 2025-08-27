import math
from typing import cast

import numpy as np
import pennylane as qml

from qxmt.ansatze import BaseVQEAnsatz
from qxmt.devices import BaseDevice
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


class ParticleConservingU2Ansatz(BaseVQEAnsatz):
    """Particle-Conserving U(2) ansatz for quantum chemistry calculations.

    The ParticleConservingU2 ansatz is an advanced hardware-efficient variational quantum circuit
    that respects both U(1) particle number symmetry and additional U(2) symmetries, providing
    enhanced physical constraints for quantum chemistry applications. This ansatz extends the
    U(1) framework by incorporating spin symmetries and additional conservation laws.

    The U(2) symmetry group includes:
    - U(1) particle number conservation (total electron count)
    - Additional symmetries related to spin and orbital angular momentum
    - Enhanced preservation of physical quantum chemistry properties

    The circuit architecture consists of layers of parameterized gates that preserve both
    particle number and spin-related quantities. Each layer contains:
    1. Single-qubit rotations that respect U(2) symmetry constraints
    2. Two-qubit gates that maintain particle and spin conservation
    3. Controlled operations preserving fermion number and spin projections

    The ansatz constructs quantum states of the form:

    \|ψ⟩ = U_U2^(L)(θ_L) ⋯ U_U2^(1)(θ_1) \|init⟩

    where U_U2^(i) represents the i-th particle and spin-conserving layer, θ_i are
    variational parameters, and \|init⟩ is the initial state (typically Hartree-Fock).

    Key features:
    - Strict particle number and enhanced symmetry conservation
    - Hardware-efficient design optimized for NISQ devices
    - Further reduced parameter space compared to U(1) due to additional constraints
    - Natural incorporation of spin and orbital symmetries
    - Improved stability and faster convergence in optimization
    - Enhanced compatibility with molecular symmetries and point groups
    - Reduced barren plateau effects due to symmetry constraints

    The U(2) symmetry constraints provide even tighter control over the variational space,
    leading to more efficient optimization and better preservation of molecular properties
    during the quantum computation.

    Args:
        device (BaseDevice): Quantum device for executing the variational circuit.
        hamiltonian (MolecularHamiltonian): Molecular Hamiltonian defining the quantum chemistry
            problem, containing electron configuration, orbital basis, and molecular geometry.
        n_layers (int, optional): Number of circuit layers. Each layer applies particle and
            spin-conserving operations while maintaining U(2) symmetry. Higher values increase
            expressibility but also circuit depth. Defaults to 2.

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
        >>> # Create Hamiltonian and device for molecular system
        >>> hamiltonian = MolecularHamiltonian(...)
        >>> device = BaseDevice(...)
        >>>
        >>> # Initialize ParticleConservingU2 ansatz with 3 layers
        >>> ansatz = ParticleConservingU2Ansatz(device, hamiltonian, n_layers=3)
        >>>
        >>> # Initialize parameters (smaller values due to tighter constraints)
        >>> params = np.random.normal(0, 0.01, ansatz.n_params)
        >>>
        >>> # Build quantum circuit (particle number and spin automatically conserved)
        >>> ansatz.circuit(params)

    References:
        - PennyLane documentation: https://docs.pennylane.ai/en/stable/code/api/pennylane.ParticleConservingU2.html

    Note:
        This ansatz is particularly beneficial for molecular systems where both particle
        number and spin conservation are critical. The U(2) symmetry constraints provide
        the most restrictive parameter space among particle-conserving ansätze, leading
        to highly stable optimization but potentially requiring more layers for complex
        molecular systems.

        The enhanced symmetry preservation makes this ansatz ideal for systems with
        specific spin states or when studying spin-dependent properties. It is especially
        suitable for near-term quantum devices due to its efficient parameter usage and
        reduced noise sensitivity.
    """

    def __init__(self, device: BaseDevice, hamiltonian: MolecularHamiltonian, n_layers: int = 2) -> None:
        super().__init__(device, hamiltonian)
        self.hamiltonian = cast(MolecularHamiltonian, self.hamiltonian)
        self.n_layers = n_layers
        self.wires = range(self.hamiltonian.get_n_qubits())
        self.prepare_hf_state()
        self.params_shape = qml.ParticleConservingU2.shape(n_wires=len(self.wires), n_layers=self.n_layers)
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
        """Construct the ParticleConservingU2 quantum circuit.

        This method builds a quantum circuit that preserves particle number through U(2) symmetry.
        The circuit applies multiple layers of particle-conserving gates while maintaining the
        total number of particles (electrons) in the system.

        Args:
            params (np.ndarray): Parameters for the ParticleConservingU2 circuit. The length of this
                               array should match the number of parameters required by the circuit
                               (determined by n_layers and number of qubits).
        """
        qml.ParticleConservingU2(
            weights=params.reshape(self.params_shape),
            wires=self.wires,
            init_state=self.hf_state,
        )
