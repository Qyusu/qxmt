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

    def test_get_energies(self) -> None:
        hamiltonian = MolecularHamiltonian(
            molname="H2",
            bondlength="0.74",
            basis_name="STO-3G",
        )

        hf_energy = hamiltonian.get_hf_energy()
        fci_energy = hamiltonian.get_fci_energy()

        assert isinstance(hf_energy, float)
        assert isinstance(fci_energy, float)
        assert fci_energy < hf_energy

    def test_set_reference_energies_with_dataset(self) -> None:
        hamiltonian = MolecularHamiltonian(
            molname="H2",
            bondlength="0.74",
            basis_name="STO-3G",
        )

        assert hasattr(hamiltonian, "hf_energy")
        assert hasattr(hamiltonian, "fci_energy")
        assert isinstance(hamiltonian.hf_energy, float)
        assert isinstance(hamiltonian.fci_energy, float)

    def test_set_reference_energies_without_dataset(self, mocker) -> None:
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        mock_compute = mocker.patch.object(MolecularHamiltonian, "_compute_energies_by_openfermionpyscf")
        mock_compute.return_value = (-1.5, -1.8)

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            hf_energy=None,
            fci_energy=None,
        )

        mock_compute.assert_called_once()

        assert hamiltonian.hf_energy == -1.5
        assert hamiltonian.fci_energy == -1.8

    def test_set_reference_energies_by_config(self) -> None:
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
        hf_energy = -1.5
        fci_energy = -1.8

        hamiltonian = MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            hf_energy=hf_energy,
            fci_energy=fci_energy,
        )

        assert hamiltonian.hf_energy == -1.5
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

    def test_compute_energies_by_openfermionpyscf(self, mocker) -> None:
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

        hf_energy, fci_energy = hamiltonian._compute_energies_by_openfermionpyscf()

        call_args = mock_run_pyscf.call_args
        assert call_args[1]["run_scf"] is True
        assert call_args[1]["run_fci"] is True
        assert call_args[1]["run_mp2"] is False
        assert call_args[1]["run_ccsd"] is False

        assert hf_energy == -1.5
        assert fci_energy == -1.8

    def test_compute_energies_by_openfermionpyscf_hf_error(self, mocker) -> None:
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        mock_compute = mocker.patch.object(MolecularHamiltonian, "_compute_energies_by_openfermionpyscf")
        mock_compute.side_effect = ValueError("HF energy is not available for the given molecule.")

        with pytest.raises(ValueError, match="HF energy is not available"):
            MolecularHamiltonian(
                symbols=symbols,
                coordinates=coordinates,
                basis_name="STO-3G",
            )

    def test_compute_energies_by_openfermionpyscf_fci_error(self, mocker) -> None:
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

        mock_compute = mocker.patch.object(MolecularHamiltonian, "_compute_energies_by_openfermionpyscf")
        mock_compute.side_effect = ValueError("FCI energy is not available for the given molecule.")

        with pytest.raises(ValueError, match="FCI energy is not available"):
            MolecularHamiltonian(
                symbols=symbols,
                coordinates=coordinates,
                basis_name="STO-3G",
            )
