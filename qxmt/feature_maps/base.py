from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pennylane as qml

from qxmt.exceptions import InputShapeError


class BaseFeatureMap(ABC):
    def __init__(self, n_qubits: int) -> None:
        self.n_qubits: int = n_qubits

    def __call__(self, x: np.ndarray) -> None:
        self.feature_map(x)

    @abstractmethod
    def feature_map(self, x: np.ndarray) -> None:
        pass

    def check_input_shape(self, x: np.ndarray, idx: int = -1) -> None:
        if x.shape[idx] != self.n_qubits:
            raise InputShapeError("Input data shape does not match the number of qubits.")

    def print_circuit(self, x: Optional[np.ndarray] = None) -> None:
        if x is None:
            x = np.random.rand(1, self.n_qubits)

        self.check_input_shape(x)
        print(qml.draw(self.feature_map)(x))
