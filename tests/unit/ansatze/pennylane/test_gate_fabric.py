import numpy as np
import pennylane as qml
import pytest
from pennylane.measurements import StateMP

from qxmt.ansatze.pennylane.gete_fabric import GeteFabricAnsatz
from qxmt.devices.pennylane_device import PennyLaneDevice
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


class TestGeteFabricAnsatz:
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
    def gate_fabric_ansatz(self, molecular_hamiltonian: MolecularHamiltonian) -> GeteFabricAnsatz:
        device = PennyLaneDevice(
            platform="pennylane",
            device_name="default.qubit",
            backend_name=None,
            n_qubits=4,
            shots=None,
        )
        return GeteFabricAnsatz(device, molecular_hamiltonian, n_layers=2, include_pi=False)

    @pytest.fixture(scope="function")
    def gate_fabric_ansatz_with_pi(self, molecular_hamiltonian: MolecularHamiltonian) -> GeteFabricAnsatz:
        device = PennyLaneDevice(
            platform="pennylane",
            device_name="default.qubit",
            backend_name=None,
            n_qubits=4,
            shots=None,
        )
        return GeteFabricAnsatz(device, molecular_hamiltonian, n_layers=3, include_pi=True)

    def test_init(self, gate_fabric_ansatz: GeteFabricAnsatz) -> None:
        assert gate_fabric_ansatz.hamiltonian is not None
        assert len(gate_fabric_ansatz.wires) == 4
        assert gate_fabric_ansatz.n_layers == 2
        assert gate_fabric_ansatz.include_pi is False
        assert hasattr(gate_fabric_ansatz, "hf_state")
        assert hasattr(gate_fabric_ansatz, "params_shape")
        assert hasattr(gate_fabric_ansatz, "n_params")

    def test_init_with_pi(self, gate_fabric_ansatz_with_pi: GeteFabricAnsatz) -> None:
        assert gate_fabric_ansatz_with_pi.n_layers == 3
        assert gate_fabric_ansatz_with_pi.include_pi is True

    def test_prepare_hf_state(self, gate_fabric_ansatz: GeteFabricAnsatz) -> None:
        gate_fabric_ansatz.prepare_hf_state()
        assert gate_fabric_ansatz.hf_state is not None
        assert len(gate_fabric_ansatz.hf_state) == 4

    def test_params_shape(self, gate_fabric_ansatz: GeteFabricAnsatz) -> None:
        assert gate_fabric_ansatz.params_shape is not None
        assert isinstance(gate_fabric_ansatz.params_shape, tuple)
        assert gate_fabric_ansatz.n_params > 0

    def test_circuit(self, gate_fabric_ansatz: GeteFabricAnsatz) -> None:
        params = np.random.rand(gate_fabric_ansatz.n_params)

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit() -> StateMP:
            gate_fabric_ansatz.circuit(params)
            return qml.state()

        state = circuit()

        assert state is not None
        assert len(state) == 2**4

        ops = [op.name for op in circuit.tape.operations]
        assert "GateFabric" in ops

    def test_circuit_with_pi(self, gate_fabric_ansatz_with_pi: GeteFabricAnsatz) -> None:
        params = np.random.rand(gate_fabric_ansatz_with_pi.n_params)

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit() -> StateMP:
            gate_fabric_ansatz_with_pi.circuit(params)
            return qml.state()

        state = circuit()

        assert state is not None
        assert len(state) == 2**4

        ops = [op.name for op in circuit.tape.operations]
        assert "GateFabric" in ops
