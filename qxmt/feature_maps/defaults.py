import numpy as np
import pennylane as qml

from qxmt.feature_maps.base import BaseFeatureMap


class SingleRotationFeatureMap(BaseFeatureMap):
    def __init__(self, n_qubits: int, reps: int, rotation_axis: list[str]) -> None:
        super().__init__(n_qubits)
        self.reps: int = reps
        self.rotation_axis: list[str] = rotation_axis

    def feature_map(
        self,
        x: np.ndarray,
    ) -> None:
        self.check_input_shape(x)
        for _ in range(self.reps):
            for ax in self.rotation_axis:
                qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation=ax)
