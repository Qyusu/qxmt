import math
from typing import cast

import numpy as np
import pennylane as qml

from qxmt.ansatze import BaseVQEAnsatz
from qxmt.devices import BaseDevice
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


class GeteFabricAnsatz(BaseVQEAnsatz):
    """GeteFabric ansatz.

    This class implements the GeteFabric ansatz for quantum chemistry calculations.
    GateFabric is a parameterized quantum circuit that uses a fabric of gates to
    prepare quantum states.

    Args:
        device (BaseDevice): Quantum device to use for the ansatz.
        hamiltonian (MolecularHamiltonian): Molecular Hamiltonian to use for the ansatz.
        n_layers (int): Number of layers in the GeteFabric circuit.
        include_pi (bool): Whether to include pi in the GeteFabric circuit.

    Attributes:
        hamiltonian (MolecularHamiltonian): Molecular Hamiltonian to use for the ansatz.
        wires (range): Wires for the GeteFabric circuit.
        n_layers (int): Number of layers in the GeteFabric circuit.
        include_pi (bool): Whether to include pi in the GeteFabric circuit.
        hf_state (np.ndarray): Hartree-Fock reference state.
        params_shape (tuple): Shape of the parameters for the GeteFabric circuit.
        n_params (int): Number of parameters for the GeteFabric circuit.

    Note:
        For optimal performance and stability with automatic differentiation,
        it is recommended to use 'parameter-shift' as the diff_method instead
        of 'adjoint' or 'best'. The GateFabric operation may not work correctly
        with adjoint differentiation in some cases.

    URL: https://docs.pennylane.ai/en/stable/code/api/pennylane.GateFabric.html
    """

    def __init__(
        self, device: BaseDevice, hamiltonian: MolecularHamiltonian, n_layers: int = 2, include_pi: bool = False
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
            electrons=self.hamiltonian.get_active_electrons(), orbitals=self.hamiltonian.get_active_spin_orbitals()
        )

    def circuit(self, params: np.ndarray) -> None:
        """Construct the GeteFabric quantum circuit.

        Args:
            params (np.ndarray): Parameters for the GeteFabric circuit. The length of this array
                               should match the number of parameters required by the circuit.
        """
        # Ensure params has the correct shape for GateFabric
        # Validate parameter count before reshaping
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(params)}")

        qml.GateFabric(
            weights=params.reshape(self.params_shape),
            wires=self.wires,
            init_state=self.hf_state,
            include_pi=self.include_pi,
        )
