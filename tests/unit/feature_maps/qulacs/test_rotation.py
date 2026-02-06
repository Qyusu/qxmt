import numpy as np
import pytest
from qulacs import QuantumCircuit

from qxmt.feature_maps.qulacs.rotation import RotationFeatureMap

N_QUBITS = 2
REPS = 1


class TestRotationFeatureMap:
    @pytest.fixture
    def rotation_feature_map(self) -> RotationFeatureMap:
        return RotationFeatureMap(n_qubits=N_QUBITS, reps=REPS, rotation_axis=["X", "Y", "Z"])

    def test_init(self, rotation_feature_map: RotationFeatureMap) -> None:
        assert rotation_feature_map.n_qubits == N_QUBITS
        assert rotation_feature_map.reps == REPS
        assert rotation_feature_map.rotation_axis == ["X", "Y", "Z"]

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="Invalid rotation axis"):
            RotationFeatureMap(n_qubits=N_QUBITS, reps=REPS, rotation_axis=["A"])

    def test_feature_map(self, rotation_feature_map: RotationFeatureMap) -> None:
        x = np.array([0.1, 0.2])
        rotation_feature_map.feature_map(x)

        assert isinstance(rotation_feature_map.circuit, QuantumCircuit)
        # 2 qubits * 3 axes = 6 gates
        gate_count = rotation_feature_map.circuit.get_gate_count()
        assert gate_count == 6

        # Check gate types
        gate_names = [rotation_feature_map.circuit.get_gate(i).get_name() for i in range(gate_count)]
        assert "X-rotation" in gate_names
        assert "Y-rotation" in gate_names
        assert "Z-rotation" in gate_names
