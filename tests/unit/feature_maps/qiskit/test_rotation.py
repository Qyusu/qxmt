import numpy as np
import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit

from qxmt.feature_maps.qiskit.rotation import HRotationFeatureMap, RotationFeatureMap

N_QUBITS = 2
REPS = 1


def operation_names(circuit: QuantumCircuit) -> list[str]:
    return [instruction.operation.name for instruction in circuit.data]


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
        assert rotation_feature_map.circuit.size() == 6
        assert operation_names(rotation_feature_map.circuit) == ["rx", "rx", "ry", "ry", "rz", "rz"]


class TestHRotationFeatureMap:
    def test_feature_map(self) -> None:
        feature_map = HRotationFeatureMap(n_qubits=N_QUBITS, reps=REPS, rotation_axis=["X", "Y"])
        x = np.array([0.1, 0.2])

        feature_map.feature_map(x)

        assert isinstance(feature_map.circuit, QuantumCircuit)
        assert feature_map.circuit.size() == 6
        assert operation_names(feature_map.circuit) == ["h", "h", "rx", "rx", "ry", "ry"]
