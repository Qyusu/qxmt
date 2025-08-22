import numpy as np
import pennylane as qml
import pytest
from pennylane.measurements import StateMP

from qxmt.ansatze.pennylane.particle_conserving_u1 import ParticleConservingU1Ansatz
from qxmt.devices.pennylane_device import PennyLaneDevice
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian


class TestParticleConservingU1Ansatz:
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
    def particle_conserving_u1_ansatz(self, molecular_hamiltonian: MolecularHamiltonian) -> ParticleConservingU1Ansatz:
        device = PennyLaneDevice(
            platform="pennylane",
            device_name="default.qubit",
            backend_name=None,
            n_qubits=4,
            shots=None,
        )
        return ParticleConservingU1Ansatz(device, molecular_hamiltonian, n_layers=2)

    @pytest.fixture(scope="function")
    def particle_conserving_u1_ansatz_deep(
        self, molecular_hamiltonian: MolecularHamiltonian
    ) -> ParticleConservingU1Ansatz:
        device = PennyLaneDevice(
            platform="pennylane",
            device_name="default.qubit",
            backend_name=None,
            n_qubits=4,
            shots=None,
        )
        return ParticleConservingU1Ansatz(device, molecular_hamiltonian, n_layers=4)

    def test_init(self, particle_conserving_u1_ansatz: ParticleConservingU1Ansatz) -> None:
        assert particle_conserving_u1_ansatz.hamiltonian is not None
        assert len(particle_conserving_u1_ansatz.wires) == 4
        assert particle_conserving_u1_ansatz.n_layers == 2
        assert hasattr(particle_conserving_u1_ansatz, "hf_state")
        assert hasattr(particle_conserving_u1_ansatz, "params_shape")
        assert hasattr(particle_conserving_u1_ansatz, "n_params")

    def test_init_deep(self, particle_conserving_u1_ansatz_deep: ParticleConservingU1Ansatz) -> None:
        assert particle_conserving_u1_ansatz_deep.n_layers == 4

    def test_prepare_hf_state(self, particle_conserving_u1_ansatz: ParticleConservingU1Ansatz) -> None:
        particle_conserving_u1_ansatz.prepare_hf_state()
        assert particle_conserving_u1_ansatz.hf_state is not None
        assert len(particle_conserving_u1_ansatz.hf_state) == 4

    def test_params_shape(self, particle_conserving_u1_ansatz: ParticleConservingU1Ansatz) -> None:
        assert particle_conserving_u1_ansatz.params_shape is not None
        assert isinstance(particle_conserving_u1_ansatz.params_shape, tuple)
        assert particle_conserving_u1_ansatz.n_params > 0

    def test_circuit(self, particle_conserving_u1_ansatz: ParticleConservingU1Ansatz) -> None:
        params = np.random.rand(particle_conserving_u1_ansatz.n_params)

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit() -> StateMP:
            particle_conserving_u1_ansatz.circuit(params)
            return qml.state()

        state = circuit()

        assert state is not None
        assert len(state) == 2**4

        ops = [op.name for op in circuit.tape.operations]
        assert "ParticleConservingU1" in ops

    def test_circuit_deep(self, particle_conserving_u1_ansatz_deep: ParticleConservingU1Ansatz) -> None:
        params = np.random.rand(particle_conserving_u1_ansatz_deep.n_params)

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit() -> StateMP:
            particle_conserving_u1_ansatz_deep.circuit(params)
            return qml.state()

        state = circuit()

        assert state is not None
        assert len(state) == 2**4

        ops = [op.name for op in circuit.tape.operations]
        assert "ParticleConservingU1" in ops
