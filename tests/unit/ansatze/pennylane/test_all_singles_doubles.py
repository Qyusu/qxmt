import numpy as np
import pennylane as qml
import pytest
from pennylane.measurements import StateMP

from qxmt.ansatze.pennylane.all_singles_doubles import AllSinglesDoublesAnsatz
from qxmt.devices.pennylane_device import PennyLaneDevice
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


class TestAllSinglesDoublesAnsatz:
    @pytest.fixture(scope="function")
    def molecular_hamiltonian(self) -> MolecularHamiltonian:
        symbols = ["H", "H"]
        coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]
        return MolecularHamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            basis_name="STO-3G",
            active_electrons=2,
            active_orbitals=2,
        )

    @pytest.fixture(scope="function")
    def all_singles_doubles_ansatz(self, molecular_hamiltonian: MolecularHamiltonian) -> AllSinglesDoublesAnsatz:
        device = PennyLaneDevice(
            platform="pennylane",
            device_name="default.qubit",
            backend_name=None,
            n_qubits=4,
            shots=None,
        )
        return AllSinglesDoublesAnsatz(device, molecular_hamiltonian)

    def test_init(self, all_singles_doubles_ansatz: AllSinglesDoublesAnsatz) -> None:
        assert all_singles_doubles_ansatz.hamiltonian is not None
        assert len(all_singles_doubles_ansatz.wires) == 4
        assert hasattr(all_singles_doubles_ansatz, "hf_state")
        assert hasattr(all_singles_doubles_ansatz, "singles")
        assert hasattr(all_singles_doubles_ansatz, "doubles")
        assert hasattr(all_singles_doubles_ansatz, "n_params")

    def test_prepare_hf_state(self, all_singles_doubles_ansatz: AllSinglesDoublesAnsatz) -> None:
        all_singles_doubles_ansatz.prepare_hf_state()
        assert all_singles_doubles_ansatz.hf_state is not None
        assert len(all_singles_doubles_ansatz.hf_state) == 4

    def test_prepare_excitation(self, all_singles_doubles_ansatz: AllSinglesDoublesAnsatz) -> None:
        assert len(all_singles_doubles_ansatz.singles) >= 0
        assert len(all_singles_doubles_ansatz.doubles) >= 0
        assert all_singles_doubles_ansatz.n_params == len(all_singles_doubles_ansatz.singles) + len(
            all_singles_doubles_ansatz.doubles
        )

    def test_circuit(self, all_singles_doubles_ansatz: AllSinglesDoublesAnsatz) -> None:
        params = np.random.rand(all_singles_doubles_ansatz.n_params)

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit() -> StateMP:
            all_singles_doubles_ansatz.circuit(params)
            return qml.state()

        state = circuit()

        assert state is not None
        assert len(state) == 2**4

        ops = [op.name for op in circuit.tape.operations]
        assert "AllSinglesDoubles" in ops
