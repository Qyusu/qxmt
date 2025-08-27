import math
from typing import cast

import numpy as np
import pennylane as qml

from qxmt.ansatze import BaseVQEAnsatz
from qxmt.devices import BaseDevice
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


class GeteFabricAnsatz(BaseVQEAnsatz):
    """Gate Fabric (GateFabric) ansatz for quantum chemistry calculations.

    The GateFabric ansatz is a hardware-efficient variational quantum circuit that employs
    a structured arrangement of parameterized gates to prepare quantum states for molecular
    ground state calculations. This ansatz is designed to balance expressibility and trainability
    while maintaining compatibility with near-term quantum hardware constraints.

    The circuit architecture consists of layers of alternating single-qubit rotations and
    two-qubit entangling gates arranged in a fabric-like pattern. Each layer applies:
    1. Parameterized single-qubit rotations (RY gates) on all qubits
    2. Controlled entangling operations between neighboring qubits
    3. Optional phase gates when include_pi is enabled

    The ansatz creates quantum states of the form:

    \|ψ⟩ = U_fabric^(L)(θ_L) ⋯ U_fabric^(1)(θ_1) \|init⟩

    where U_fabric^(i) represents the i-th layer, θ_i are variational parameters,
    and \|init⟩ is the initial state (typically Hartree-Fock).

    Key features:
    - Hardware-efficient design suitable for NISQ devices
    - Structured gate arrangement optimizing qubit connectivity
    - Adjustable circuit depth through layer parameter
    - Built-in initialization from Hartree-Fock state
    - Support for phase gates to enhance expressibility
    - Polynomial parameter scaling with system size

    Args:
        device (BaseDevice): Quantum device for executing the variational circuit.
        hamiltonian (MolecularHamiltonian): Molecular Hamiltonian defining the quantum chemistry
            problem, containing electron, orbital, and geometry information.
        n_layers (int, optional): Number of circuit layers. Higher values increase expressibility
            but also circuit depth and parameter count. Defaults to 2.
        include_pi (bool, optional): Whether to include additional phase (π) gates in the circuit.
            This can enhance expressibility at the cost of increased complexity. Defaults to False.

    Attributes:
        hamiltonian (MolecularHamiltonian): Molecular Hamiltonian for the ansatz.
        wires (range): Qubit indices used in the quantum circuit.
        n_layers (int): Number of layers in the circuit architecture.
        include_pi (bool): Flag indicating whether π gates are included.
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
        >>> # Initialize GateFabric ansatz with 3 layers and π gates
        >>> ansatz = GeteFabricAnsatz(device, hamiltonian, n_layers=3, include_pi=True)
        >>>
        >>> # Initialize parameters
        >>> params = np.random.normal(0, 0.1, ansatz.n_params)
        >>>
        >>> # Build quantum circuit
        >>> ansatz.circuit(params)

    References:
        - PennyLane documentation: https://docs.pennylane.ai/en/stable/code/api/pennylane.GateFabric.html

    Note:
        For optimal performance and stability with automatic differentiation,
        it is recommended to use 'parameter-shift' as the diff_method instead
        of 'adjoint' or 'best'. The GateFabric operation may not work correctly
        with adjoint differentiation in some cases due to its complex gate structure.

        This ansatz is particularly useful when chemical intuition about excitations is limited
        or when exploring hardware-efficient implementations for specific quantum devices.
    """

    def __init__(
        self,
        device: BaseDevice,
        hamiltonian: MolecularHamiltonian,
        n_layers: int = 2,
        include_pi: bool = False,
    ) -> None:
        super().__init__(device, hamiltonian)
        self.hamiltonian = cast(MolecularHamiltonian, self.hamiltonian)
        self.wires = range(self.hamiltonian.get_n_qubits())
        self.n_layers = n_layers
        self.include_pi = include_pi
        self.prepare_hf_state()
        self.params_shape = qml.GateFabric.shape(n_layers=n_layers, n_wires=len(self.wires))
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
        """Construct the GeteFabric quantum circuit.

        Args:
            params (np.ndarray): Parameters for the GeteFabric circuit. The length of this array
                               should match the number of parameters required by the circuit.
        """
        qml.GateFabric(
            weights=params.reshape(self.params_shape),
            wires=self.wires,
            init_state=self.hf_state,
            include_pi=self.include_pi,
        )
