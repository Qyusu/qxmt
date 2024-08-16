import numpy as np
import pennylane as qml

from qxmt.feature_maps.base import BaseFeatureMap

PENNYLANE_PLATFORM: str = "pennylane"


class RotationFeatureMap(BaseFeatureMap):
    def __init__(self, n_qubits: int, reps: int, rotation_axis: list[str]) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
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


class HRotationFeatureMap(BaseFeatureMap):
    def __init__(self, n_qubits: int, reps: int, rotation_axis: list[str]) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.reps: int = reps
        self.rotation_axis: list[str] = rotation_axis

    def feature_map(self, x: np.ndarray) -> None:
        self.check_input_shape(x)
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            for ax in self.rotation_axis:
                qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation=ax)


class XXFeatureMap(BaseFeatureMap):
    def __init__(self, n_qubits: int, reps: int) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.reps: int = reps

    def feature_map(self, x: np.ndarray) -> None:
        self.check_input_shape(x)
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                qml.RX(x[i], wires=i)
            for i in range(0, self.n_qubits - 1):
                qml.IsingXX(2 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=[i, i + 1])


class YYFeatureMap(BaseFeatureMap):
    def __init__(self, n_qubits: int, reps: int) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.reps: int = reps

    def feature_map(self, x: np.ndarray) -> None:
        self.check_input_shape(x)
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(x[i], wires=i)
            for i in range(0, self.n_qubits - 1):
                qml.IsingYY(2 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=[i, i + 1])


class ZZFeatureMap(BaseFeatureMap):
    def __init__(self, n_qubits: int, reps: int) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.reps: int = reps

    def feature_map(self, x: np.ndarray) -> None:
        self.check_input_shape(x)
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x[i], wires=i)
            for i in range(0, self.n_qubits - 1):
                qml.IsingZZ(2 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=[i, i + 1])
