import numpy as np
import pennylane as qml
import pytest
from pennylane.measurements import StateMP

from qxmt.feature_maps.pennylane.yzcx import YZCXFeatureMap

N_QUBITS = 2


class TestYZCXFeatureMap:
    @pytest.fixture(scope="function")
    def yzcx_feature_map(self) -> YZCXFeatureMap:
        # create YZCX Feature Map instance
        feature_map = YZCXFeatureMap(n_qubits=N_QUBITS, reps=2, c=1.0, seed=42)
        return feature_map

    def test_init(self, yzcx_feature_map: YZCXFeatureMap) -> None:
        assert yzcx_feature_map.n_qubits == N_QUBITS
        assert yzcx_feature_map.reps == 2
        assert yzcx_feature_map.c == 1.0
        assert yzcx_feature_map.seed == 42

    def test_feature_map(self, yzcx_feature_map: YZCXFeatureMap) -> None:
        # create a quantum device and circuit
        input_data = np.array([0.5, 0.3, 0.8, 0.2])
        dev = qml.device("default.qubit", wires=N_QUBITS)

        @qml.qnode(dev)
        def circuit() -> StateMP:
            yzcx_feature_map.feature_map(input_data)
            return qml.state()

        circuit()

        # check if the expected gates are in the circuit
        ops = [op.name for op in circuit.tape.operations]
        expected_gates = ["RY", "RZ", "CNOT"]
        for expected_gate in expected_gates:
            assert any(expected_gate in op for op in ops)
