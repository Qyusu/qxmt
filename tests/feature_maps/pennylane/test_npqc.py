import numpy as np
import pennylane as qml
import pytest
from pennylane.measurements import StateMP

from qxmt.feature_maps.pennylane.npqc import NPQCFeatureMap

DEFAULT_N_QUBITS = 2


class TestNPQCFeatureMap:
    @pytest.fixture(scope="function")
    def npqc_feature_map(self) -> NPQCFeatureMap:
        # create NPQC Feature Map instance
        feature_map = NPQCFeatureMap(n_qubits=DEFAULT_N_QUBITS, reps=2, c=1.0)
        return feature_map

    def test_init(self, npqc_feature_map: NPQCFeatureMap) -> None:
        assert npqc_feature_map.n_qubits == DEFAULT_N_QUBITS
        assert npqc_feature_map.reps == 2
        assert npqc_feature_map.c == 1.0

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="NPQC feature map requires an even number of qubits. but got 3"):
            NPQCFeatureMap(n_qubits=3, reps=2, c=1.0)

    def test_feature_map(self, npqc_feature_map: NPQCFeatureMap) -> None:
        # create a quantum device and circuit
        input_data = np.array([0.5, 0.3, 0.8, 0.2])
        dev = qml.device("default.qubit", wires=DEFAULT_N_QUBITS)

        @qml.qnode(dev)
        def circuit() -> StateMP:
            npqc_feature_map.feature_map(input_data)
            return qml.state()

        circuit()

        # check if the expected gates are in the circuit
        ops = [op.name for op in circuit.tape.operations]
        expected_gates = ["RY", "RZ", "CZ"]
        for expected_gate in expected_gates:
            assert any(expected_gate in op for op in ops)
