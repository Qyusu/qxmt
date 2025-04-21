import numpy as np
import pytest

from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


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
