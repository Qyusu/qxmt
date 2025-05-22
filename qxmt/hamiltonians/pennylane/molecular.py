from enum import Enum, auto
from logging import Logger
from typing import Literal, Optional

import pennylane as qml
from pennylane import numpy as qnp
from pennylane.ops.op_math import Sum

from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.hamiltonians import BaseHamiltonian
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)

DATA_MODULE_NAME = "qchem"
SUPPORTED_BASIS_NAMES = Literal["STO-3G", "6-31G", "6-311G", "CC-PVDZ"]
SUPPORTED_UNITS = Literal["angstrom", "bohr"]
SUPPORTED_METHODS = Literal["dhf", "pyscf", "openfermion"]
SUPPORTED_MAPPINGS = Literal["jordan_wigner", "bravyi_kitaev", "parity"]


class InitializationType(Enum):
    DATASET = auto()
    DIRECT_MOLECULE = auto()
    INVALID = auto()


class MolecularHamiltonian(BaseHamiltonian):
    """Molecular Hamiltonian for quantum chemistry calculations.

    This class represents a molecular Hamiltonian using PennyLane's quantum chemistry module.
    It supports both full and active space calculations for molecular systems.

    Args:
        molname: Name of the molecule in PennyLane's dataset.
        bondlength: Bond length of the molecule in Angstroms.
        symbols: List of atomic symbols (e.g., ['H', 'H'] for H2).
        coordinates: Array of atomic coordinates in Angstroms.
        charge: Total charge of the molecule. Defaults to 0.
        multi: Multiplicity of the molecule. Defaults to 1.
        basis_name: Basis set name. Only supported ["sto-3g", "6-31g", "6-311g", "cc-pvdz"]. Defaults to "sto-3g".
        unit: Unit of the coordinates. Only supported ["angstrom", "bohr"]. Defaults to "angstrom".
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
        molname: Optional[str] = None,
        bondlength: Optional[float | str] = None,
        symbols: Optional[list[str]] = None,
        coordinates: Optional[qnp.tensor | list[float] | list[list[float]]] = None,
        charge: int = 0,
        multi: int = 1,
        basis_name: SUPPORTED_BASIS_NAMES = "STO-3G",
        unit: SUPPORTED_UNITS = "angstrom",
        method: SUPPORTED_METHODS = "dhf",
        active_electrons: Optional[int] = None,
        active_orbitals: Optional[int] = None,
        mapping: SUPPORTED_MAPPINGS = "jordan_wigner",
        logger: Logger = LOGGER,
    ) -> None:
        super().__init__(platform=PENNYLANE_PLATFORM)

        self.molname: Optional[str] = molname
        self.bondlength: Optional[float | str] = bondlength
        self.symbols: Optional[list[str]] = symbols
        self.coordinates: Optional[qnp.tensor | list[float] | list[list[float]]] = (
            qnp.array(coordinates, requires_grad=False) if isinstance(coordinates, list) else coordinates
        )  # type: ignore
        self.charge: int = charge
        self.multi: int = multi
        self.basis_name: SUPPORTED_BASIS_NAMES = basis_name
        self.unit: SUPPORTED_UNITS = unit
        self.method: SUPPORTED_METHODS = method
        self.active_electrons: Optional[int] = active_electrons
        self.active_orbitals: Optional[int] = active_orbitals
        self.mapping: SUPPORTED_MAPPINGS = mapping
        self.logger: Logger = logger
        self.hamiltonian: Sum
        self.n_qubits: int
        self.molecule: qml.qchem.Molecule
        self.hf_energy: float
        self.fci_energy: Optional[float] = None
        self._dataset: list = []

        self._initialize_hamiltonian()
        self._set_hf_energy()
        self._set_fci_energy()

    def _determine_initialization_type(self) -> InitializationType:
        """Determine the initialization type based on the provided parameters.

        Returns:
            InitializationType: The type of initialization.
        """
        if self.molname is not None and self.bondlength is not None:
            return InitializationType.DATASET
        elif self.symbols is not None and self.coordinates is not None:
            return InitializationType.DIRECT_MOLECULE
        return InitializationType.INVALID

    def _validate_bondlength(self) -> None:
        """Validate the bond length for the specified molecule and basis set.

        This method:
        1. Retrieves the list of valid bond lengths for the given molecule and basis set
           from PennyLane's dataset
        2. Checks if the provided bond length is in the list of valid values
        3. Raises ValueError if the bond length is not valid

        Raises:
            ValueError: If the provided bond length is not in the list of valid values
                       for the given molecule and basis set.
        """
        valid_bondlengths = qml.data.list_datasets()[DATA_MODULE_NAME][self.molname][self.basis_name]
        if str(self.bondlength) not in valid_bondlengths:
            raise ValueError(f"Invalid bondlength: {self.bondlength}. Valid bondlengths are {valid_bondlengths}")

    def _initialize_hamiltonian(self) -> None:
        """Initialize the molecular Hamiltonian.

        This method:
        1. Creates a Molecule object either from:
           - A predefined dataset (if molname and bondlength are provided)
           - Atomic symbols and coordinates (if symbols and coordinates are provided)
        2. Constructs the molecular Hamiltonian using PennyLane's quantum chemistry module
        3. Sets the number of qubits required for the simulation
        4. Raises ValueError if neither dataset nor atomic information is provided
        """
        init_type = self._determine_initialization_type()
        if init_type == InitializationType.DATASET:
            self._validate_bondlength()
            # force=False: if the dataset already exists, it will not be loaded again
            self._dataset = qml.data.load(
                DATA_MODULE_NAME, molname=self.molname, basis=self.basis_name, bondlength=self.bondlength, force=False
            )
            self.molecule = self._dataset[0].molecule
        elif init_type == InitializationType.DIRECT_MOLECULE:
            self.molecule = qml.qchem.Molecule(
                self.symbols,
                self.coordinates,
                charge=self.charge,
                mult=self.multi,
                basis_name=self.basis_name,
                unit=self.unit,
            )
        else:
            raise ValueError('Either "molname" and "bondlength" or "symbols" and "coordinates" must be provided')

        hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(
            molecule=self.molecule,
            method=self.method,
            active_electrons=self.active_electrons,
            active_orbitals=self.active_orbitals,
            mapping=self.mapping,
        )
        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits

    def _set_hf_energy(self) -> None:
        """Set the Hartree-Fock energy of the molecule.

        Returns:
            float: Hartree-Fock energy.
        """
        self.hf_energy = float(qml.qchem.hf_energy(self.molecule)())

    def _set_fci_energy(self) -> None:
        """Set the FCI energy of the molecule.
        FCI energy is only available for molecules in the dataset.
        If the molecule is directly specified, FCI energy is not available.

        [TODO]: FCI energy get from pyscf

        Returns:
            float: FCI energy.
        """
        if self.molname is not None:
            if not self._dataset:
                raise ValueError("Dataset is not loaded. Please load the dataset first.")
            self.fci_energy = float(self._dataset[0].fci_energy)
        else:
            self.logger.warning("FCI energy is not available for the given molecule. Setting to None.")
            self.fci_energy = None

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

    def get_hf_energy(self) -> float:
        """Get the Hartree-Fock energy of the molecule.

        Returns:
            float: Hartree-Fock energy.
        """
        return self.hf_energy

    def get_fci_energy(self) -> Optional[float]:
        """Get the FCI energy of the molecule.

        Returns:
            float: FCI energy.
        """
        return self.fci_energy
