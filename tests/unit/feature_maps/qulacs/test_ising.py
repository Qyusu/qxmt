import numpy as np
import pytest

pytest.importorskip("qulacs")

from qulacs import QuantumCircuit

from qxmt.feature_maps.qulacs.ising import XXFeatureMap, YYFeatureMap, ZZFeatureMap

N_QUBITS = 2
REPS = 2


class TestIsingFeatureMap:
    @pytest.fixture(scope="function")
    def xx_feature_map(self) -> XXFeatureMap:
        return XXFeatureMap(n_qubits=N_QUBITS, reps=REPS)

    @pytest.fixture(scope="function")
    def yy_feature_map(self) -> YYFeatureMap:
        return YYFeatureMap(n_qubits=N_QUBITS, reps=REPS)

    @pytest.fixture(scope="function")
    def zz_feature_map(self) -> ZZFeatureMap:
        return ZZFeatureMap(n_qubits=N_QUBITS, reps=REPS)

    @pytest.mark.parametrize(
        "feature_map_fixture, expected_n_qubits, expected_reps",
        [
            ("xx_feature_map", N_QUBITS, REPS),
            ("yy_feature_map", N_QUBITS, REPS),
            ("zz_feature_map", N_QUBITS, REPS),
        ],
    )
    def test_init(
        self,
        request: pytest.FixtureRequest,
        feature_map_fixture: str,
        expected_n_qubits: int,
        expected_reps: int,
    ) -> None:
        feature_map = request.getfixturevalue(feature_map_fixture)
        assert feature_map.n_qubits == expected_n_qubits
        assert feature_map.reps == expected_reps

    def test_xx_feature_map_circuit(self, xx_feature_map: XXFeatureMap) -> None:
        x = np.array([0.1, 0.2])
        xx_feature_map.feature_map(x)

        assert isinstance(xx_feature_map.circuit, QuantumCircuit)
        # 2 qubits * 2 rep -> (RX * 4) + (MultiPauli * 2) = 6 gates
        gate_count = xx_feature_map.circuit.get_gate_count()
        assert gate_count == 6

        gate_names = [xx_feature_map.circuit.get_gate(i).get_name() for i in range(gate_count)]
        assert "X-rotation" in gate_names
        assert "Pauli-rotation" in gate_names

    def test_yy_feature_map_circuit(self, yy_feature_map: YYFeatureMap) -> None:
        x = np.array([0.1, 0.2])
        yy_feature_map.feature_map(x)

        assert isinstance(yy_feature_map.circuit, QuantumCircuit)
        # 2 qubits * 1 rep -> (H * 4 + RY * 4) + (MultiPauli * 2) = 10 gates
        gate_count = yy_feature_map.circuit.get_gate_count()
        assert gate_count == 10

        gate_names = [yy_feature_map.circuit.get_gate(i).get_name() for i in range(gate_count)]
        assert "H" in gate_names
        assert "Y-rotation" in gate_names
        assert "Pauli-rotation" in gate_names

    def test_zz_feature_map_circuit(self, zz_feature_map: ZZFeatureMap) -> None:
        x = np.array([0.1, 0.2])
        zz_feature_map.feature_map(x)

        assert isinstance(zz_feature_map.circuit, QuantumCircuit)
        # 2 qubits * 1 rep -> (H * 4 + RZ * 4) + (MultiPauli * 2) = 10 gates
        gate_count = zz_feature_map.circuit.get_gate_count()
        assert gate_count == 10

        gate_names = [zz_feature_map.circuit.get_gate(i).get_name() for i in range(gate_count)]
        assert "H" in gate_names
        assert "Z-rotation" in gate_names
        assert "Pauli-rotation" in gate_names
