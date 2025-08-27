from enum import Enum, auto
from logging import Logger
from typing import Literal, Optional

import pennylane as qml
from openfermion.chem.molecular_data import MolecularData
from openfermionpyscf import run_pyscf
from pennylane import numpy as qnp
from pennylane.ops.op_math import Sum

from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.hamiltonians import BaseHamiltonian
from qxmt.hamiltonians.energy_data import ReferenceEnergies
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)

DATA_MODULE_NAME = "qchem"
SUPPORTED_BASIS_NAMES = Literal["STO-3G", "6-31G", "6-311G", "CC-PVDZ"]
SUPPORTED_UNITS = Literal["angstrom", "bohr"]
SUPPORTED_METHODS = Literal["dhf", "pyscf", "openfermion"]
SUPPORTED_MAPPINGS = Literal["jordan_wigner", "bravyi_kitaev", "parity"]
SUPPORTED_REFERENCE_ENERGY_METHOD = Literal["casci", "casscf", "fci"]


class InitializationType(Enum):
    DATASET = auto()
    DIRECT_MOLECULE = auto()
    INVALID = auto()


class MolecularHamiltonian(BaseHamiltonian):
    """Molecular Hamiltonian for quantum chemistry calculations.

    This class represents a molecular Hamiltonian using PennyLane's quantum chemistry module.
    It supports both full and active space calculations for molecular systems, including CASCI
    (Complete Active Space Configuration Interaction) and CASSCF (Complete Active Space
    Self-Consistent Field) with frozen core orbitals.

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
        hf_energy: Pre-computed HF energy. If None, will be computed.
        fci_energy: Pre-computed FCI energy. If None, will be computed.
        use_casci: Whether to use CASCI instead of FCI for energy calculations. Defaults to False.
        cas_type: Type of CAS calculation ("casci" or "casscf"). Defaults to "casci".
        max_cycle_macro: Maximum number of macro iterations for CASSCF. Defaults to 50.
        conv_tol: Convergence tolerance for CASSCF. Defaults to 1e-7.
        orbital_optimization: Whether to perform orbital optimization in CASSCF. Defaults to True.
        logger: Logger instance for logging.

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
        hf_energy: Optional[float] = None,
        casci_energy: Optional[float] = None,
        casscf_energy: Optional[float] = None,
        fci_energy: Optional[float] = None,
        reference_energy_methods: list[SUPPORTED_REFERENCE_ENERGY_METHOD] = ["fci"],
        max_cycle_macro: int = 50,
        conv_tol: float = 1e-7,
        orbital_optimization: bool = True,
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
        self.reference_energy_methods: list[SUPPORTED_REFERENCE_ENERGY_METHOD] = reference_energy_methods
        self.max_cycle_macro: int = int(max_cycle_macro)
        self.conv_tol: float = float(conv_tol)
        self.orbital_optimization: bool = bool(orbital_optimization)
        self.logger: Logger = logger
        self.hamiltonian: Sum
        self.n_qubits: int
        self.molecule: qml.qchem.Molecule
        self.reference_energies: ReferenceEnergies = ReferenceEnergies()
        self._dataset: list = []

        self._initialize_hamiltonian()
        self.reference_energies = self._get_reference_energies(
            ReferenceEnergies(
                hf_energy=hf_energy,
                casci_energy=casci_energy,
                casscf_energy=casscf_energy,
                fci_energy=fci_energy,
            )
        )

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

    def _get_reference_energies(
        self,
        init_energies: ReferenceEnergies,
    ) -> ReferenceEnergies:
        """Get reference energies based on the specified methods.

        Args:
            init_energies: Initial energy values. If None, will be computed.

        Returns:
            ReferenceEnergies: Object containing all computed reference energies.

        Computes energies for all methods specified in reference_energy_methods.
        For dataset molecules, HF and FCI are retrieved from cache.
        For custom molecules, energies are computed using PySCF.
        """
        result_energies = ReferenceEnergies(
            hf_energy=init_energies.hf_energy,
            casci_energy=init_energies.casci_energy,
            casscf_energy=init_energies.casscf_energy,
            fci_energy=init_energies.fci_energy,
        )

        if self.molname is not None:
            if not self._dataset:
                raise ValueError("Dataset is not loaded. Please load the dataset first.")

            # For dataset molecules, get HF and FCI from cache
            result_energies.hf_energy = float(qml.qchem.hf_energy(self.molecule)())
            result_energies.fci_energy = float(self._dataset[0].fci_energy)
            return result_energies

        else:
            if result_energies.hf_energy is None:
                scf_result = run_pyscf(
                    self._pennylane_molecule2openfermion(),
                    run_scf=True,
                    run_fci=False,
                    run_mp2=False,
                    run_ccsd=False,
                )
                result_energies.hf_energy = float(scf_result.hf_energy)  # type: ignore

            # Compute requested reference energies
            openfermion_molecule = self._pennylane_molecule2openfermion()

            for method in self.reference_energy_methods:
                if method in ["casci", "casscf"]:
                    # Check if active space parameters are available
                    if self.active_orbitals is None or self.active_electrons is None:
                        self.logger.warning(
                            f"Skipping {method.upper()} calculation: active_orbitals and active_electrons must be specified"
                        )
                        continue

                if method == "casci" and result_energies.casci_energy is None:
                    assert self.active_orbitals is not None and self.active_electrons is not None
                    result_energies.casci_energy = self._run_cas_calculation(
                        openfermion_molecule, self.active_orbitals, self.active_electrons, calculation_type="casci"
                    )
                    self.logger.debug(f"Computed CASCI energy: {result_energies.casci_energy}")

                elif method == "casscf" and result_energies.casscf_energy is None:
                    assert self.active_orbitals is not None and self.active_electrons is not None
                    result_energies.casscf_energy = self._run_cas_calculation(
                        openfermion_molecule, self.active_orbitals, self.active_electrons, calculation_type="casscf"
                    )
                    self.logger.debug(f"Computed CASSCF energy: {result_energies.casscf_energy}")

                elif method == "fci" and result_energies.fci_energy is None:
                    fci_result = run_pyscf(
                        openfermion_molecule,
                        run_scf=True,
                        run_fci=True,
                        run_mp2=False,
                        run_ccsd=False,
                    )
                    result_energies.fci_energy = float(fci_result.fci_energy)  # type: ignore
                    self.logger.debug(f"Computed FCI energy: {result_energies.fci_energy}")

        return result_energies

    def _pennylane_molecule2openfermion(self) -> MolecularData:
        """Convert the PennyLane molecule to OpenFermion molecule."""
        geometry = list(zip(self.symbols, self.coordinates.tolist()))  # type: ignore
        return MolecularData(geometry=geometry, basis=self.basis_name, multiplicity=self.multi, charge=self.charge)

    def _run_cas_calculation(
        self, openfermion_molecule: MolecularData, cas_norb: int, cas_nelec: int, calculation_type: str
    ) -> float:
        """Run CASCI or CASSCF calculation with frozen core using PySCF.

        Args:
            openfermion_molecule: OpenFermion MolecularData object.
            cas_norb: Number of active orbitals for CAS calculation.
            cas_nelec: Number of active electrons for CAS calculation.
            calculation_type: Type of calculation ("casci" or "casscf").

        Returns:
            CASCI or CASSCF energy.
        """
        try:
            import pyscf
            from pyscf import mcscf
        except ImportError:
            raise ImportError(
                "PySCF is required for CASCI/CASSCF calculations. Please install it with 'pip install pyscf'"
            )

        # Build molecule in PySCF format
        mol = pyscf.gto.Mole()
        mol.atom = []

        # Convert geometry from OpenFermion format to PySCF format
        for atom_symbol, coords in openfermion_molecule.geometry:
            mol.atom.append([atom_symbol, coords])

        mol.basis = openfermion_molecule.basis
        mol.charge = openfermion_molecule.charge
        mol.spin = openfermion_molecule.multiplicity - 1  # PySCF uses 2S instead of 2S+1
        mol.build()

        # Run HF calculation first to get molecular orbitals
        mf = pyscf.scf.RHF(mol)
        mf.kernel()

        # Determine frozen core orbitals
        # Typically, we freeze the core orbitals (1s for atoms beyond H/He)
        n_frozen_core = self._determine_frozen_core_orbitals(mol)

        # Set up CAS calculation with frozen core
        if calculation_type.lower() == "casscf":
            # Use CASSCF with orbital optimization
            if n_frozen_core > 0:
                ncore = n_frozen_core
                ncas = cas_norb
                nelec_cas = cas_nelec
                cas_solver = mcscf.CASSCF(mf, ncas, nelec_cas)
                setattr(cas_solver, "ncore", ncore)  # type: ignore
            else:
                cas_solver = mcscf.CASSCF(mf, cas_norb, cas_nelec)

            # Set CASSCF parameters
            max_cycle_val = int(self.max_cycle_macro)
            conv_tol_val = float(self.conv_tol)

            self.logger.debug(
                f"Setting CASSCF parameters: max_cycle_macro={max_cycle_val} (type: {type(max_cycle_val)}), conv_tol={conv_tol_val} (type: {type(conv_tol_val)})"
            )

            setattr(cas_solver, "max_cycle_macro", max_cycle_val)  # type: ignore
            setattr(cas_solver, "conv_tol", conv_tol_val)  # type: ignore

            # Disable orbital optimization if specified
            if not self.orbital_optimization:
                setattr(cas_solver, "max_cycle_macro", int(1))  # type: ignore
                setattr(cas_solver, "frozen", True)  # type: ignore

        else:
            # Use CASCI (no orbital optimization)
            if n_frozen_core > 0:
                ncore = n_frozen_core
                ncas = cas_norb
                nelec_cas = cas_nelec
                cas_solver = mcscf.CASCI(mf, ncas, nelec_cas)
                setattr(cas_solver, "ncore", ncore)  # type: ignore
            else:
                cas_solver = mcscf.CASCI(mf, cas_norb, cas_nelec)

        # Run CAS calculation
        cas_energy = cas_solver.kernel()[0]  # type: ignore

        self.logger.debug(f"{calculation_type.upper()} calculation completed with {n_frozen_core} frozen core orbitals")
        self.logger.debug(f"Active space: {cas_nelec} electrons in {cas_norb} orbitals")
        if calculation_type.lower() == "casscf":
            converged = getattr(cas_solver, "converged", None)  # type: ignore
            self.logger.debug(f"CASSCF converged: {converged}")

        return float(cas_energy)

    def _determine_frozen_core_orbitals(self, mol) -> int:
        """Determine the number of frozen core orbitals based on the molecular composition.

        Args:
            mol: PySCF molecule object.

        Returns:
            Number of frozen core orbitals.
        """
        n_frozen = 0

        for atom_idx in range(mol.natm):
            atomic_number = mol.atom_charge(atom_idx)

            # Freeze 1s orbitals for atoms with Z > 2 (beyond He)
            if atomic_number > 2:
                n_frozen += 1

            # For heavier atoms, could freeze more core orbitals (This is a simple heuristic)
            if atomic_number > 10:  # Beyond Ne
                n_frozen += 1  # Additional core orbitals (2s, 2p)

        return n_frozen

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

    def get_hf_energy(self) -> Optional[float]:
        """Get the Hartree-Fock energy of the molecule.

        Returns:
            float: Hartree-Fock energy.
        """
        return self.reference_energies.hf_energy

    def get_casci_energy(self) -> Optional[float]:
        """Get the CASCI energy of the molecule.

        Returns:
            float: CASCI energy.
        """
        return self.reference_energies.casci_energy

    def get_casscf_energy(self) -> Optional[float]:
        """Get the CASSCF energy of the molecule.

        Returns:
            float: CASSCF energy.
        """
        return self.reference_energies.casscf_energy

    def get_fci_energy(self) -> Optional[float]:
        """Get the FCI energy of the molecule.

        Returns:
            float: FCI energy.
        """
        return self.reference_energies.fci_energy

    def get_reference_energies(self) -> ReferenceEnergies:
        """Get the complete reference energies object.

        Returns:
            ReferenceEnergies: Object containing all computed reference energies.
        """
        return self.reference_energies
