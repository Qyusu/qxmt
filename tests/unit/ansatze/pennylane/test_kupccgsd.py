import numpy as np
import pennylane as qml
import pytest
from pennylane.measurements import StateMP

from qxmt.ansatze.pennylane.kupccgsd import KUpCCGSDAnsatz
from qxmt.devices.pennylane_device import PennyLaneDevice
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


class TestKUpCCGSDAnsatz:
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
    def kupccgsd_ansatz(self, molecular_hamiltonian: MolecularHamiltonian) -> KUpCCGSDAnsatz:
        device = PennyLaneDevice(
            platform="pennylane",
            device_name="default.qubit",
            backend_name=None,
            n_qubits=4,
            shots=None,
        )
        return KUpCCGSDAnsatz(device, molecular_hamiltonian, k=1, delta_sz=0)

    @pytest.fixture(scope="function")
    def kupccgsd_ansatz_k2(self, molecular_hamiltonian: MolecularHamiltonian) -> KUpCCGSDAnsatz:
        device = PennyLaneDevice(
            platform="pennylane",
            device_name="default.qubit",
            backend_name=None,
            n_qubits=4,
            shots=None,
        )
        return KUpCCGSDAnsatz(device, molecular_hamiltonian, k=2, delta_sz=1)

    def test_init(self, kupccgsd_ansatz: KUpCCGSDAnsatz) -> None:
        assert kupccgsd_ansatz.hamiltonian is not None
        assert len(kupccgsd_ansatz.wires) == 4
        assert kupccgsd_ansatz.k == 1
        assert kupccgsd_ansatz.delta_sz == 0
        assert hasattr(kupccgsd_ansatz, "hf_state")
        assert hasattr(kupccgsd_ansatz, "params_shape")
        assert hasattr(kupccgsd_ansatz, "n_params")

    def test_init_k2(self, kupccgsd_ansatz_k2: KUpCCGSDAnsatz) -> None:
        assert kupccgsd_ansatz_k2.k == 2
        assert kupccgsd_ansatz_k2.delta_sz == 1

    def test_prepare_hf_state(self, kupccgsd_ansatz: KUpCCGSDAnsatz) -> None:
        kupccgsd_ansatz.prepare_hf_state()
        assert kupccgsd_ansatz.hf_state is not None
        assert len(kupccgsd_ansatz.hf_state) == 4

    def test_params_shape(self, kupccgsd_ansatz: KUpCCGSDAnsatz) -> None:
        assert kupccgsd_ansatz.params_shape is not None
        assert isinstance(kupccgsd_ansatz.params_shape, tuple)
        assert kupccgsd_ansatz.n_params > 0

    def test_circuit(self, kupccgsd_ansatz: KUpCCGSDAnsatz) -> None:
        params = np.random.rand(kupccgsd_ansatz.n_params)

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit() -> StateMP:
            kupccgsd_ansatz.circuit(params)
            return qml.state()

        state = circuit()

        assert state is not None
        assert len(state) == 2**4

        ops = [op.name for op in circuit.tape.operations]
        assert "kUpCCGSD" in ops

    def test_circuit_k2(self, kupccgsd_ansatz_k2: KUpCCGSDAnsatz) -> None:
        params = np.random.rand(kupccgsd_ansatz_k2.n_params)

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit() -> StateMP:
            kupccgsd_ansatz_k2.circuit(params)
            return qml.state()

        state = circuit()

        assert state is not None
        assert len(state) == 2**4

        ops = [op.name for op in circuit.tape.operations]
        assert "kUpCCGSD" in ops
