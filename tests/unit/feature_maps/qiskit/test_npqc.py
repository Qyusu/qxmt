import numpy as np
import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit

from qxmt.feature_maps.qiskit.npqc import NPQCFeatureMap

N_QUBITS = 2
REPS = 2


class TestNPQCFeatureMap:
    @pytest.fixture
    def npqc_feature_map(self) -> NPQCFeatureMap:
        return NPQCFeatureMap(n_qubits=N_QUBITS, reps=REPS, c=1.0)

    def test_init(self, npqc_feature_map: NPQCFeatureMap) -> None:
        assert npqc_feature_map.n_qubits == N_QUBITS
        assert npqc_feature_map.reps == REPS
        assert npqc_feature_map.c == 1.0

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="NPQC feature map requires an even number of qubits"):
            NPQCFeatureMap(n_qubits=3, reps=1, c=1.0)

    def test_feature_map(self, npqc_feature_map: NPQCFeatureMap) -> None:
        x = np.array([0.1, 0.2, 0.3, 0.4])
        npqc_feature_map.feature_map(x)

        assert isinstance(npqc_feature_map.circuit, QuantumCircuit)
        assert npqc_feature_map.circuit.size() == 11

        gate_names = [instruction.operation.name for instruction in npqc_feature_map.circuit.data]
        assert "ry" in gate_names
        assert "rz" in gate_names
        assert "cz" in gate_names
