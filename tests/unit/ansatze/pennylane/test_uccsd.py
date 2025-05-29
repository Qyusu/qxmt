import numpy as np
import pennylane as qml
import pytest
from pennylane.measurements import StateMP

from qxmt.ansatze.pennylane.uccsd import UCCSDAnsatz
from qxmt.devices.pennylane_device import PennyLaneDevice
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


class TestUCCSDAnsatz:
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
    def uccsd_ansatz(self, molecular_hamiltonian: MolecularHamiltonian) -> UCCSDAnsatz:
        device = PennyLaneDevice(
            platform="pennylane",
            device_name="default.qubit",
            backend_name=None,
            n_qubits=4,
            shots=None,
        )
        return UCCSDAnsatz(device, molecular_hamiltonian)

    def test_init(self, uccsd_ansatz: UCCSDAnsatz) -> None:
        assert uccsd_ansatz.hamiltonian is not None
        assert len(uccsd_ansatz.wires) == 4
        assert hasattr(uccsd_ansatz, "hf_state")
        assert hasattr(uccsd_ansatz, "singles")
        assert hasattr(uccsd_ansatz, "doubles")
        assert hasattr(uccsd_ansatz, "s_wires")
        assert hasattr(uccsd_ansatz, "d_wires")

    def test_prepare_hf_state(self, uccsd_ansatz: UCCSDAnsatz) -> None:
        uccsd_ansatz.prepare_hf_state()
        assert uccsd_ansatz.hf_state is not None
        assert len(uccsd_ansatz.hf_state) == 4

    def test_prepare_excitation_wires(self, uccsd_ansatz: UCCSDAnsatz) -> None:
        assert len(uccsd_ansatz.singles) >= 0
        assert len(uccsd_ansatz.doubles) >= 0
        assert len(uccsd_ansatz.s_wires) >= 0
        assert len(uccsd_ansatz.d_wires) >= 0

    def test_circuit(self, uccsd_ansatz: UCCSDAnsatz) -> None:
        n_params = len(uccsd_ansatz.singles) + len(uccsd_ansatz.doubles)
        params = np.random.rand(n_params)

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit() -> StateMP:
            uccsd_ansatz.circuit(params)
            return qml.state()

        state = circuit()

        assert state is not None
        assert len(state) == 2**4

        ops = [op.name for op in circuit.tape.operations]
        assert "UCCSD" in ops
