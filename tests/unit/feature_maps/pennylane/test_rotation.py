import numpy as np
import pennylane as qml
import pytest
from pennylane.measurements import StateMP

from qxmt.feature_maps.pennylane.rotation import HRotationFeatureMap, RotationFeatureMap

N_QUBITS = 2


class TestRotationFeatureMap:
    @pytest.fixture(scope="function")
    def rotation_feature_map(self) -> RotationFeatureMap:
        feature_map = RotationFeatureMap(n_qubits=N_QUBITS, reps=1, rotation_axis=["X"])
        return feature_map

    @pytest.fixture(scope="function")
    def h_rotation_feature_map(self) -> HRotationFeatureMap:
        feature_map = HRotationFeatureMap(n_qubits=N_QUBITS, reps=1, rotation_axis=["X"])
        return feature_map

    @pytest.mark.parametrize(
        "feature_map_name, expected_n_qubits, expected_reps, expected_rotation_axis",
        [
            pytest.param("rotation_feature_map", N_QUBITS, 1, ["X"], id="Valid RotationFeatureMap"),
            pytest.param("h_rotation_feature_map", N_QUBITS, 1, ["X"], id="Valid HRotationFeatureMap"),
        ],
    )
    def test_init(
        self,
        request: pytest.FixtureRequest,
        feature_map_name: str,
        expected_n_qubits: int,
        expected_reps: int,
        expected_rotation_axis: list[str],
    ) -> None:
        feature_map = request.getfixturevalue(feature_map_name)
        assert feature_map.n_qubits == expected_n_qubits
        assert feature_map.reps == expected_reps
        assert feature_map.rotation_axis == expected_rotation_axis

    @pytest.mark.parametrize(
        "feature_map_name, input_data, expected_gates",
        [
            pytest.param(
                "rotation_feature_map", np.array([0.5, 0.3]), ["AngleEmbedding"], id="Valid RotationFeatureMap"
            ),
            pytest.param(
                "h_rotation_feature_map",
                np.array([0.5, 0.3]),
                ["Hadamard", "AngleEmbedding"],
                id="Valid HRotationFeatureMap",
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
            feature_map.feature_map(input_data)
            return qml.state()

        circuit()

        # check if the expected gates are in the circuit
        ops = [op.name for op in circuit.tape.operations]
        for expected_gate in expected_gates:
            assert any(expected_gate in op for op in ops)
