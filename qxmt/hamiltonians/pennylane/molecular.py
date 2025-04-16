from typing import Optional

import numpy as np
import pennylane as qml
from pennylane.ops.op_math import Sum

from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.hamiltonians import BaseHamiltonian


class MolecularHamiltonian(BaseHamiltonian):
    """Molecular Hamiltonian for quantum chemistry calculations.

    This class represents a molecular Hamiltonian using PennyLane's quantum chemistry module.
    It supports both full and active space calculations for molecular systems.

    Args:
        symbols: List of atomic symbols (e.g., ['H', 'H'] for H2).
        coordinates: Array of atomic coordinates in Angstroms.
        charge: Total charge of the molecule. Defaults to 0.
        multi: Multiplicity of the molecule. Defaults to 1.
        basis_name: Basis set name. Only supported ["sto-3g", "6-31g", "6-311g", "cc-pvdz"]. Defaults to "sto-3g".
        method: Method to use for the calculation. Only supported ["dhf", "pyscf", "openfermion"]. Defaults to "dhf".
        active_electrons: Number of active electrons. If None, uses all electrons.
        active_orbitals: Number of active orbitals. If None, uses all orbitals.
        mapping: Mapping to use for the calculation. Defaults to "jordan_wigner".

    Attributes:
        hamiltonian: PennyLane Hamiltonian operator.
        n_qubits: Number of qubits required for the simulation.
        molecule: PennyLane Molecule object.
    """

    def __init__(
        self,
        symbols: list[str],
        coordinates: np.ndarray | list[float],
        charge: int = 0,
        multi: int = 1,
        basis_name: str = "sto-3g",
        method: str = "dhf",
        active_electrons: Optional[int] = None,
        active_orbitals: Optional[int] = None,
        mapping: str = "jordan_wigner",
    ) -> None:
        super().__init__(platform=PENNYLANE_PLATFORM)
        self.symbols: list[str] = symbols
        self.coordinates: np.ndarray = np.array(coordinates)
        self.charge: int = charge
        self.multi: int = multi
        self.basis_name: str = basis_name
        self.method: str = method
        self.active_electrons: Optional[int] = active_electrons
        self.active_orbitals: Optional[int] = active_orbitals
        self.mapping: str = mapping
        self.hamiltonian: Sum
        self.n_qubits: int
        self.molecule: qml.qchem.Molecule
        self._initialize_hamiltonian()

    def _initialize_hamiltonian(self) -> None:
        """Initialize the molecular Hamiltonian.

        This method:
        1. Creates a Molecule object from the atomic symbols and coordinates
        2. Constructs the molecular Hamiltonian using PennyLane's quantum chemistry module
        3. Sets the number of qubits required for the simulation
        """
        self.molecule = qml.qchem.Molecule(
            self.symbols,
            self.coordinates,
            charge=self.charge,
            mult=self.multi,
            basis_name=self.basis_name,
        )
        hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(
            molecule=self.molecule,
            method=self.method,
            active_electrons=self.active_electrons,
            active_orbitals=self.active_orbitals,
            mapping=self.mapping,
        )
        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits

    def get_hamiltonian(self) -> Sum:
        """Get the Hamiltonian operator.

        Returns:
            Sum: The molecular Hamiltonian operator.
        """
        return self.hamiltonian

    def get_n_qubits(self) -> int:
        """Get the number of qubits required for the simulation.

        Returns:
            int: Number of qubits.
        """
        return self.n_qubits

    def get_molecule(self) -> qml.qchem.Molecule:
        """Get the Molecule object.

        Returns:
            qml.qchem.Molecule: The molecule object.
        """
        return self.molecule

    def get_electrons(self) -> int:
        """Get the total number of electrons in the molecule.

        Returns:
            int: Total number of electrons.
        """
        return self.molecule.n_electrons

    def get_molecular_orbitals(self) -> int:
        """Get the number of molecular orbitals.

        Returns:
            int: Number of molecular orbitals.
        """
        return self.molecule.n_orbitals

    def get_spin_orbitals(self) -> int:
        """Get the number of spin orbitals.

        Returns:
            int: Number of spin orbitals (2 * number of molecular orbitals).
        """
        return 2 * self.molecule.n_orbitals

    def get_active_electrons(self) -> int:
        """Get the number of active electrons.

        If active_electrons is not specified, returns the total number of electrons.

        Returns:
            int: Number of active electrons.
        """
        if self.active_electrons is None:
            return self.get_electrons()
        else:
            return self.active_electrons

    def get_active_orbitals(self) -> int:
        """Get the number of active orbitals.

        If active_orbitals is not specified, returns the total number of molecular orbitals.

        Returns:
            int: Number of active orbitals.
        """
        if self.active_orbitals is None:
            return self.get_molecular_orbitals()
        else:
            return self.active_orbitals

    def get_active_spin_orbitals(self) -> int:
        """Get the number of active spin orbitals.

        If active_orbitals is not specified, returns the total number of spin orbitals.

        Returns:
            int: Number of active spin orbitals (2 * number of active orbitals).
        """
        if self.active_orbitals is None:
            return self.get_spin_orbitals()
        else:
            return 2 * self.active_orbitals
