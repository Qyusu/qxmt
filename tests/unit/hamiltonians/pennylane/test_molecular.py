import numpy as np
import pytest
from pytest_mock import MockerFixture

from qxmt.hamiltonians.energy_data import ReferenceEnergies
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


@pytest.fixture
def init_energies() -> ReferenceEnergies:
    return ReferenceEnergies(
        hf_energy=None,
        casci_energy=None,
        casscf_energy=None,
        fci_energy=None,
    )


# Molecule dataset cannot access simultaneously in parallel
@pytest.mark.serial
class TestMolecularHamiltonian:
    def test_initialize_with_dataset(self) -> None:
        hamiltonian = MolecularHamiltonian(
            molname="H2",
            bondlength="0.74",
            basis_name="STO-3G",
        )

        assert hamiltonian.molname == "H2"
        assert hamiltonian.bondlength == "0.74"
        assert hamiltonian.basis_name == "STO-3G"
        assert hamiltonian.n_qubits > 0
        assert hamiltonian.hamiltonian is not None

    def test_initialize_with_atoms(self) -> None:
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
        )

        assert hamiltonian.symbols == symbols
        assert np.allclose(hamiltonian.coordinates, coordinates)
        assert hamiltonian.n_qubits > 0
        assert hamiltonian.hamiltonian is not None

    def test_invalid_initialization(self) -> None:
        # missing required parameters
        with pytest.raises(ValueError):
            MolecularHamiltonian()

        # invalid bondlength
        with pytest.raises(ValueError):
            MolecularHamiltonian(
                molname="H2",
                bondlength="invalid_length",
                basis_name="STO-3G",
            )

    def test_get_molecule_properties(self) -> None:
        hamiltonian = MolecularHamiltonian(
            molname="H2",
            bondlength="0.74",
            basis_name="STO-3G",
        )

        assert hamiltonian.get_electrons() > 0
        assert hamiltonian.get_molecular_orbitals() > 0
        assert hamiltonian.get_spin_orbitals() == 2 * hamiltonian.get_molecular_orbitals()

    def test_active_space(self) -> None:
        hamiltonian = MolecularHamiltonian(
            molname="H2",
            bondlength="0.74",
            basis_name="STO-3G",
            active_electrons=2,
            active_orbitals=2,
        )

        assert hamiltonian.get_active_electrons() == 2
        assert hamiltonian.get_active_orbitals() == 2
        assert hamiltonian.get_active_spin_orbitals() == 4

    def test_set_reference_energies_with_dataset(self) -> None:
        hamiltonian = MolecularHamiltonian(
            molname="H2",
            bondlength="0.74",
            basis_name="STO-3G",
        )

        assert hasattr(hamiltonian, "reference_energies")
        assert isinstance(hamiltonian.reference_energies.hf_energy, float)
        assert hamiltonian.reference_energies.casci_energy is None
        assert isinstance(hamiltonian.reference_energies.fci_energy, float)

    def test_get_energies(self) -> None:
        hamiltonian = MolecularHamiltonian(
            molname="H2",
            bondlength="0.74",
            basis_name="STO-3G",
        )

        hf_energy = hamiltonian.get_hf_energy()
        casci_energy = hamiltonian.get_casci_energy()
        fci_energy = hamiltonian.get_fci_energy()

        assert isinstance(hf_energy, float)
        assert casci_energy is None
        assert isinstance(fci_energy, float)
        assert fci_energy < hf_energy

    def test_reference_energy_methods_fci_only(self) -> None:
        """Test FCI-only calculation."""
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        # Use real PySCF-backed computation

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            reference_energy_methods=["fci"],
        )

        assert isinstance(hamiltonian.reference_energies.hf_energy, float)
        assert hamiltonian.reference_energies.casci_energy is None
        assert hamiltonian.reference_energies.casscf_energy is None
        assert isinstance(hamiltonian.reference_energies.fci_energy, float)
        assert hamiltonian.reference_energies.fci_energy < hamiltonian.reference_energies.hf_energy

    def test_reference_energy_methods_casci_only(self) -> None:
        """Test CASCI-only calculation."""
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            active_electrons=2,
            active_orbitals=2,
            reference_energy_methods=["casci"],
        )

        assert isinstance(hamiltonian.reference_energies.hf_energy, float)
        assert isinstance(hamiltonian.reference_energies.casci_energy, float)
        assert hamiltonian.reference_energies.casscf_energy is None
        assert hamiltonian.reference_energies.fci_energy is None
        assert hamiltonian.reference_energies.casci_energy < hamiltonian.reference_energies.hf_energy

    def test_set_reference_energies_by_config(self) -> None:
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
        hf_energy = -1.5
        casci_energy = -1.6
        fci_energy = -1.8

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            hf_energy=hf_energy,
            casci_energy=casci_energy,
            fci_energy=fci_energy,
        )

        assert hamiltonian.reference_energies.hf_energy == -1.5
        assert hamiltonian.reference_energies.casci_energy == -1.6
        assert hamiltonian.reference_energies.fci_energy == -1.8

    def test_get_reference_energies_without_dataset_error(self, init_energies: ReferenceEnergies) -> None:
        hamiltonian = MolecularHamiltonian(
            molname="H2",
            bondlength="0.74",
            basis_name="STO-3G",
        )

        hamiltonian._dataset = []

        with pytest.raises(ValueError, match="Dataset is not loaded"):
            hamiltonian._get_reference_energies(init_energies)

    def test_build_pyscf_molecule(self) -> None:
        pyscf = pytest.importorskip("pyscf")
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            charge=0,
            multi=1,
        )

        mol = hamiltonian._build_pyscf_molecule()

        assert isinstance(mol, pyscf.gto.Mole)
        assert mol.charge == 0
        assert mol.spin == 0
        assert mol.natm == 2

    def test_reference_energy_methods_multiple(self) -> None:
        """Test multiple reference energy methods calculation."""
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        # Use real PySCF-backed computations

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            active_electrons=2,
            active_orbitals=2,
            reference_energy_methods=["casci", "casscf", "fci"],
        )

        assert isinstance(hamiltonian.reference_energies.hf_energy, float)
        assert isinstance(hamiltonian.reference_energies.casci_energy, float)
        assert isinstance(hamiltonian.reference_energies.casscf_energy, float)
        assert isinstance(hamiltonian.reference_energies.fci_energy, float)
        assert hamiltonian.reference_energies.casci_energy <= hamiltonian.reference_energies.hf_energy
        assert hamiltonian.reference_energies.casscf_energy <= hamiltonian.reference_energies.hf_energy
        assert hamiltonian.reference_energies.fci_energy <= hamiltonian.reference_energies.casscf_energy

    def test_reference_energy_methods_missing_active_space(self) -> None:
        """Test behavior when active space parameters are missing for CAS methods."""
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            reference_energy_methods=["casci", "casscf"],
            # Note: No active_electrons or active_orbitals specified
        )

        # Verify CAS calculations were skipped (energies remain None)
        assert isinstance(hamiltonian.reference_energies.hf_energy, float)
        assert hamiltonian.reference_energies.casci_energy is None
        assert hamiltonian.reference_energies.casscf_energy is None
        assert hamiltonian.reference_energies.fci_energy is None

    def test_reference_energy_methods_default_fci(self, mocker) -> None:
        """Test default reference energy method (FCI)."""
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        # Mock internal methods (default FCI)
        mocker.patch.object(MolecularHamiltonian, "_build_pyscf_molecule", return_value=mocker.MagicMock())
        mocker.patch.object(MolecularHamiltonian, "_compute_hf_energy_by_pyscf", return_value=-1.1)
        mocker.patch.object(MolecularHamiltonian, "_compute_fci_energy_by_pyscf", return_value=-1.15)

        # Default should be FCI only
        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            reference_energy_methods=["fci"],
        )

        assert isinstance(hamiltonian.reference_energies.hf_energy, float)
        assert hamiltonian.reference_energies.casci_energy is None
        assert isinstance(hamiltonian.reference_energies.fci_energy, float)
        assert hamiltonian.reference_energies.fci_energy < hamiltonian.reference_energies.hf_energy

    def test_determine_frozen_core_orbitals(self, mocker) -> None:
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
        )

        # Mock PySCF molecule
        mock_mol = mocker.MagicMock()
        mock_mol.natm = 2
        mock_mol.atom_charge.side_effect = [1, 1]  # H atoms with Z=1

        n_frozen = hamiltonian._determine_frozen_core_orbitals(mock_mol)
        assert n_frozen == 0  # No core orbitals for H atoms

        # Test with heavier atoms
        mock_mol.natm = 2
        mock_mol.atom_charge.side_effect = [6, 8]  # C and O atoms

        n_frozen = hamiltonian._determine_frozen_core_orbitals(mock_mol)
        assert n_frozen == 2  # 1s orbitals for both C and O

        # Test with very heavy atoms
        mock_mol.natm = 1
        mock_mol.atom_charge.side_effect = [12]  # Mg atom

        n_frozen = hamiltonian._determine_frozen_core_orbitals(mock_mol)
        assert n_frozen == 2  # 1s and additional core orbitals

    def test_compute_cas_energy_inconsistent_electrons_raises(self, mocker) -> None:
        """Ensure CAS calculation validates electron count parity/positivity."""
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            active_electrons=2,
            active_orbitals=2,
        )

        # Odd difference (2 - 1) -> raises
        with pytest.raises(ValueError, match="non-negative even number"):
            hamiltonian._compute_cas_energy_by_pyscf(cas_norb=1, cas_nelec=1, calculation_type="casci")

        # Negative difference (2 - 3) -> raises
        with pytest.raises(ValueError, match="non-negative even number"):
            hamiltonian._compute_cas_energy_by_pyscf(cas_norb=1, cas_nelec=3, calculation_type="casci")

    def test_casci_integration(self, mocker) -> None:
        """Test complete CASCI workflow from initialization to energy calculation."""
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            active_electrons=2,
            active_orbitals=2,
            reference_energy_methods=["casci"],
        )

        hf_energy = hamiltonian.get_hf_energy()
        cas_energy = hamiltonian.get_casci_energy()
        fci_energy = hamiltonian.get_fci_energy()
        assert isinstance(hf_energy, float)
        assert isinstance(cas_energy, float)
        assert cas_energy <= hf_energy
        assert fci_energy is None

    def test_reference_energy_methods_parameter_validation(self) -> None:
        """Test that reference_energy_methods parameter is properly validated."""
        symbols = ["Li", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        # Test with FCI method
        hamiltonian_fci = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            active_electrons=2,
            active_orbitals=5,
            reference_energy_methods=["fci"],
        )

        assert isinstance(hamiltonian_fci.get_hf_energy(), float)
        assert isinstance(hamiltonian_fci.get_fci_energy(), float)
        assert hamiltonian_fci.get_casci_energy() is None
        assert hamiltonian_fci.get_casscf_energy() is None

        # Test with multiple methods
        hamiltonian_multi = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            active_electrons=2,
            active_orbitals=5,
            reference_energy_methods=["casci", "casscf", "fci"],
        )

        hf_energy = hamiltonian_multi.get_hf_energy()
        casci_energy = hamiltonian_multi.get_casci_energy()
        casscf_energy = hamiltonian_multi.get_casscf_energy()
        fci_energy = hamiltonian_multi.get_fci_energy()
        assert isinstance(hf_energy, float)
        assert isinstance(casci_energy, float)
        assert isinstance(casscf_energy, float)
        assert isinstance(fci_energy, float)
        assert fci_energy <= casscf_energy <= casci_energy <= hf_energy

    def test_casscf_call_and_result(self) -> None:
        """Test CASSCF calculation path and returned energy via real PySCF."""
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            active_electrons=2,
            active_orbitals=2,
            reference_energy_methods=["casscf"],
            max_cycle_macro=30,
            conv_tol=1e-8,
        )

        hf_energy = hamiltonian.get_hf_energy()
        cas_energy = hamiltonian.get_casscf_energy()
        assert hf_energy is not None and isinstance(hf_energy, float)
        assert cas_energy is not None and isinstance(cas_energy, float)
        assert cas_energy <= hf_energy

    def test_casscf_integration(self, mocker: MockerFixture) -> None:
        """Test complete CASSCF workflow from initialization to energy calculation."""
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        # Mock internal HF and CASSCF calculations
        mocker.patch.object(MolecularHamiltonian, "_build_pyscf_molecule", return_value=mocker.MagicMock())
        mocker.patch.object(MolecularHamiltonian, "_compute_hf_energy_by_pyscf", return_value=-1.5)
        mock_cas_calc = mocker.patch.object(MolecularHamiltonian, "_compute_cas_energy_by_pyscf", return_value=-1.65)

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            active_electrons=2,
            active_orbitals=2,
            reference_energy_methods=["casscf"],
        )

        assert hamiltonian.get_hf_energy() == -1.5
        assert hamiltonian.get_casci_energy() is None
        assert hamiltonian.get_casscf_energy() == -1.65
        assert hamiltonian.get_fci_energy() is None

        mock_cas_calc.assert_called_once()

    def test_reference_energies_object(self) -> None:
        """Test ReferenceEnergies object functionality."""
        hamiltonian = MolecularHamiltonian(
            molname="H2",
            bondlength="0.74",
            basis_name="STO-3G",
        )

        # Test that reference_energies object exists
        assert hasattr(hamiltonian, "reference_energies")
        assert isinstance(hamiltonian.reference_energies, ReferenceEnergies)

        # Test get_reference_energies method
        energies = hamiltonian.get_reference_energies()
        assert isinstance(energies, ReferenceEnergies)
