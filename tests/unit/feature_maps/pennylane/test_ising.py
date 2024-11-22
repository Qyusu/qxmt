import numpy as np
import pennylane as qml
import pytest
from pennylane.measurements import StateMP

from qxmt.feature_maps.pennylane.ising import XXFeatureMap, YYFeatureMap, ZZFeatureMap

N_QUBITS = 2


class TestIsingFeatureMap:
    @pytest.fixture(scope="function")
    def xx_feature_map(self) -> XXFeatureMap:
        # create XX Feature Map instance
        feature_map = XXFeatureMap(n_qubits=N_QUBITS, reps=2)
        return feature_map

    @pytest.fixture(scope="function")
    def yy_feature_map(self) -> YYFeatureMap:
        # create YY Feature Map instance
        feature_map = YYFeatureMap(n_qubits=N_QUBITS, reps=2)
        return feature_map

    @pytest.fixture(scope="function")
    def zz_feature_map(self) -> ZZFeatureMap:
        # create ZZ Feature Map instance
        feature_map = ZZFeatureMap(n_qubits=N_QUBITS, reps=2)
        return feature_map

    @pytest.mark.parametrize(
        "feature_map_name, expected_n_qubits, expected_reps",
        [
            pytest.param("xx_feature_map", N_QUBITS, 2, id="Valid XXFeatureMap"),
            pytest.param("yy_feature_map", N_QUBITS, 2, id="Valid YYFeatureMap"),
            pytest.param("zz_feature_map", N_QUBITS, 2, id="Valid ZZFeatureMap"),
        ],
    )
    def test_init(
        self,
        request: pytest.FixtureRequest,
        feature_map_name: str,
        expected_n_qubits: int,
        expected_reps: int,
    ) -> None:
        feature_map = request.getfixturevalue(feature_map_name)
        assert feature_map.n_qubits == expected_n_qubits
        assert feature_map.reps == expected_reps

    @pytest.mark.parametrize(
        "feature_map_name, input_data, expected_gates",
        [
            pytest.param("xx_feature_map", np.array([0.5, 0.3]), ["RX", "IsingXX"], id="Valid XXFeatureMap"),
            pytest.param(
                "yy_feature_map", np.array([0.5, 0.3]), ["Hadamard", "RY", "IsingYY"], id="Valid YYFeatureMap"
            ),
            pytest.param(
                "zz_feature_map", np.array([0.5, 0.3]), ["Hadamard", "RZ", "IsingZZ"], id="Valid ZZFeatureMap"
            ),
        ],
    )
    def test_feature_map(
        self,
        request: pytest.FixtureRequest,
        feature_map_name: str,
        input_data: np.ndarray,
        expected_gates: list[str],
    ) -> None:
        feature_map = request.getfixturevalue(feature_map_name)

        # create a quantum device and circuit
        dev = qml.device("default.qubit", wires=N_QUBITS)

        @qml.qnode(dev)
        def circuit() -> StateMP:
            feature_map(input_data)
            return qml.state()

        circuit()

        # check if the expected gates are in the circuit
        ops = [op.name for op in circuit.tape.operations]
        for expected_gate in expected_gates:
            assert any(expected_gate in op for op in ops)
