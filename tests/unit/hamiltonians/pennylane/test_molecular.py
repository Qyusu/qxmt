import numpy as np
import pytest
from openfermion.chem.molecular_data import MolecularData

from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


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

        assert hasattr(hamiltonian, "hf_energy")
        assert hasattr(hamiltonian, "casci_energy")
        assert hasattr(hamiltonian, "fci_energy")
        assert isinstance(hamiltonian.hf_energy, float)
        assert hamiltonian.casci_energy is None
        assert isinstance(hamiltonian.fci_energy, float)

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

    def test_set_reference_energies_without_dataset(self, mocker) -> None:
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        # test without casci
        mock_compute = mocker.patch.object(MolecularHamiltonian, "_compute_energies_by_pyscf")
        mock_compute.return_value = (-1.5, None, -1.8)

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            hf_energy=None,
            casci_energy=None,
            fci_energy=None,
            use_casci=False,
        )

        mock_compute.assert_called_once_with(use_casci=False)

        assert hamiltonian.hf_energy == -1.5
        assert hamiltonian.casci_energy is None
        assert hamiltonian.fci_energy == -1.8

        # test with casci
        mock_compute = mocker.patch.object(MolecularHamiltonian, "_compute_energies_by_pyscf")
        mock_compute.return_value = (-1.5, -1.6, None)

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            active_electrons=2,
            active_orbitals=2,
            hf_energy=None,
            casci_energy=None,
            fci_energy=None,
            use_casci=True,
        )

        mock_compute.assert_called_once_with(use_casci=True)

        assert hamiltonian.hf_energy == -1.5
        assert hamiltonian.casci_energy == -1.6
        assert hamiltonian.fci_energy is None

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

        assert hamiltonian.hf_energy == -1.5
        assert hamiltonian.casci_energy == -1.6
        assert hamiltonian.fci_energy == -1.8

    def test_get_reference_energies_without_dataset_error(self) -> None:
        hamiltonian = MolecularHamiltonian(
            molname="H2",
            bondlength="0.74",
            basis_name="STO-3G",
        )

        hamiltonian._dataset = []

        with pytest.raises(ValueError, match="Dataset is not loaded"):
            hamiltonian._get_reference_energies()

    def test_pennylane_molecule2openfermion(self) -> None:
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            charge=0,
            multi=1,
        )

        openfermion_molecule = hamiltonian._pennylane_molecule2openfermion()

        assert isinstance(openfermion_molecule, MolecularData)
        assert openfermion_molecule.basis == "STO-3G"
        assert openfermion_molecule.multiplicity == 1
        assert openfermion_molecule.charge == 0

    def test_compute_energies_by_pyscf_fci(self, mocker) -> None:
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        mock_energy = mocker.MagicMock()
        mock_energy.hf_energy = -1.5
        mock_energy.fci_energy = -1.8
        mock_run_pyscf = mocker.patch("qxmt.hamiltonians.pennylane.molecular.run_pyscf")
        mock_run_pyscf.return_value = mock_energy

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
        )

        hf_energy, casci_energy, fci_energy = hamiltonian._compute_energies_by_pyscf(use_casci=False)

        call_args = mock_run_pyscf.call_args
        assert call_args[1]["run_scf"] is True
        assert call_args[1]["run_fci"] is True
        assert call_args[1]["run_mp2"] is False
        assert call_args[1]["run_ccsd"] is False

        assert hf_energy == -1.5
        assert casci_energy is None
        assert fci_energy == -1.8

    def test_compute_energies_by_pyscf_casci(self, mocker) -> None:
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        mock_energy = mocker.MagicMock()
        mock_energy.hf_energy = -1.5
        mock_run_pyscf = mocker.patch("qxmt.hamiltonians.pennylane.molecular.run_pyscf")
        mock_run_pyscf.return_value = mock_energy

        mock_casci = mocker.patch.object(MolecularHamiltonian, "_run_casci_with_frozen_core")
        mock_casci.return_value = -1.6

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            active_electrons=2,
            active_orbitals=2,
        )

        hf_energy, casci_energy, fci_energy = hamiltonian._compute_energies_by_pyscf(use_casci=True)

        call_args = mock_run_pyscf.call_args
        assert call_args[1]["run_scf"] is True
        assert call_args[1]["run_fci"] is False
        assert call_args[1]["run_mp2"] is False
        assert call_args[1]["run_ccsd"] is False

        mock_casci.assert_called_once()

        assert hf_energy == -1.5
        assert casci_energy == -1.6
        assert fci_energy is None

    def test_compute_energies_by_pyscf_casci_missing_parameters(self) -> None:
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
        )

        with pytest.raises(ValueError, match="active_orbitals and active_electrons must be specified"):
            hamiltonian._compute_energies_by_pyscf(use_casci=True)

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

    def test_run_casci_with_frozen_core(self, mocker) -> None:
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            active_electrons=2,
            active_orbitals=2,
        )

        mock_pyscf = mocker.MagicMock()
        mock_mcscf = mocker.MagicMock()
        mock_gto = mocker.MagicMock()
        mock_scf = mocker.MagicMock()

        mocker.patch.dict("sys.modules", {"pyscf": mock_pyscf, "pyscf.mcscf": mock_mcscf})
        mock_pyscf.gto = mock_gto
        mock_pyscf.scf = mock_scf
        mock_pyscf.mcscf = mock_mcscf

        mock_mol = mocker.MagicMock()
        mock_gto.Mole.return_value = mock_mol

        mock_mf = mocker.MagicMock()
        mock_scf.RHF.return_value = mock_mf

        mock_casci = mocker.MagicMock()
        mock_casci.kernel.return_value = (-1.6, None)  # Return energy and other data
        mock_mcscf.CASCI.return_value = mock_casci

        mock_determine_frozen = mocker.patch.object(hamiltonian, "_determine_frozen_core_orbitals")
        mock_determine_frozen.return_value = 0  # No frozen core for H2

        openfermion_molecule = hamiltonian._pennylane_molecule2openfermion()

        casci_energy = hamiltonian._run_casci_with_frozen_core(openfermion_molecule, 2, 2)

        assert casci_energy == -1.6
        mock_mol.build.assert_called_once()
        mock_mf.kernel.assert_called_once()
        mock_casci.kernel.assert_called_once()

    def test_run_casci_with_frozen_core_import_error(self, mocker) -> None:
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            active_electrons=2,
            active_orbitals=2,
        )

        original_import = __builtins__["__import__"]

        def mock_import(name, *args):
            if name == "pyscf":
                raise ImportError("PySCF not available")
            return original_import(name, *args)

        mocker.patch("builtins.__import__", side_effect=mock_import)

        openfermion_molecule = hamiltonian._pennylane_molecule2openfermion()

        with pytest.raises(ImportError, match="PySCF is required for CASCI calculations"):
            hamiltonian._run_casci_with_frozen_core(openfermion_molecule, 2, 2)

    def test_casci_integration(self, mocker) -> None:
        """Test complete CASCI workflow from initialization to energy calculation."""
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        mock_compute = mocker.patch.object(MolecularHamiltonian, "_compute_energies_by_pyscf")
        mock_compute.return_value = (-1.5, -1.6, None)

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            active_electrons=2,
            active_orbitals=2,
            use_casci=True,
        )

        assert hamiltonian.get_hf_energy() == -1.5
        assert hamiltonian.get_casci_energy() == -1.6
        assert hamiltonian.get_fci_energy() is None

        mock_compute.assert_called_once_with(use_casci=True)

    def test_use_casci_parameter_propagation(self, mocker) -> None:
        """Test that use_casci parameter is properly propagated through the call chain."""
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        mock_compute = mocker.patch.object(MolecularHamiltonian, "_compute_energies_by_pyscf")
        mock_compute.return_value = (-1.5, -1.6, None)

        # Test with use_casci=True
        hamiltonian_casci = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            active_electrons=2,
            active_orbitals=2,
            use_casci=True,
        )

        call_args = mock_compute.call_args
        assert call_args[1]["use_casci"] is True

        assert hamiltonian_casci.get_hf_energy() == -1.5
        assert hamiltonian_casci.get_casci_energy() == -1.6
        assert hamiltonian_casci.get_fci_energy() is None

        # Reset mock for second test
        mock_compute.reset_mock()
        mock_compute.return_value = (-1.5, None, -1.8)

        # Test with use_casci=False (default)
        hamiltonian_fci = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            use_casci=False,
        )

        call_args = mock_compute.call_args
        assert call_args[1]["use_casci"] is False

        assert hamiltonian_fci.get_hf_energy() == -1.5
        assert hamiltonian_fci.get_casci_energy() is None
        assert hamiltonian_fci.get_fci_energy() == -1.8
