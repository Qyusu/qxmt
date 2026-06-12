import numpy as np
import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit

from qxmt.feature_maps.qiskit.yzcx import YZCXFeatureMap

N_QUBITS = 2
REPS = 2


class TestYZCXFeatureMap:
    @pytest.fixture
    def yzcx_feature_map(self) -> YZCXFeatureMap:
        return YZCXFeatureMap(n_qubits=N_QUBITS, reps=REPS, c=1.0, seed=42)

    def test_init(self, yzcx_feature_map: YZCXFeatureMap) -> None:
        assert yzcx_feature_map.n_qubits == N_QUBITS
        assert yzcx_feature_map.reps == REPS
        assert yzcx_feature_map.c == 1.0
        assert yzcx_feature_map.seed == 42

    def test_feature_map(self, yzcx_feature_map: YZCXFeatureMap) -> None:
        x = np.array([0.1, 0.2, 0.3, 0.4])
        yzcx_feature_map.feature_map(x)

        assert isinstance(yzcx_feature_map.circuit, QuantumCircuit)
        assert yzcx_feature_map.circuit.size() == 17

        gate_names = [instruction.operation.name for instruction in yzcx_feature_map.circuit.data]
        assert "ry" in gate_names
        assert "rz" in gate_names
        assert "cx" in gate_names
