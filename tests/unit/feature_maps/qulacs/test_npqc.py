import numpy as np
import pytest
from qulacs import QuantumCircuit

from qxmt.feature_maps.qulacs.npqc import NPQCFeatureMap

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

        gate_count = npqc_feature_map.circuit.get_gate_count()
        assert gate_count == 11

        gate_names = [npqc_feature_map.circuit.get_gate(i).get_name() for i in range(gate_count)]
        assert "Y-rotation" in gate_names
        assert "Z-rotation" in gate_names
        assert "CZ" in gate_names
