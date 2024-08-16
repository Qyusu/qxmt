from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pennylane as qml

from qxmt.constants import SUPPORTED_PLATFORMS
from qxmt.exceptions import InputShapeError, InvalidPlatformError


class BaseFeatureMap(ABC):
    def __init__(self, platform: str, n_qubits: int) -> None:
        self.platform: str = platform
        self.n_qubits: int = n_qubits

        if self.platform not in SUPPORTED_PLATFORMS:
            raise InvalidPlatformError(
                f"Platform '{self.platform}' is not supported. Please choose in {SUPPORTED_PLATFORMS}."
            )

    def __call__(self, x: np.ndarray) -> None:
        self.feature_map(x)

    @abstractmethod
    def feature_map(self, x: np.ndarray) -> None:
        """Feature map function that maps input data to quantum states.

        Args:
            x (np.ndarray): input data
        """
        pass

    def check_input_shape(self, x: np.ndarray, idx: int = -1) -> None:
        """Check if the input data shape matches the number of qubits.

        Args:
            x (np.ndarray): input data
            idx (int, optional): index of the dimension of qubit. Defaults to -1.

        Raises:
            InputShapeError: input data shape does not match the number of qubits.
        """
        if x.shape[idx] != self.n_qubits:
            raise InputShapeError("Input data shape does not match the number of qubits.")

    def print_circuit(self, x: Optional[np.ndarray] = None) -> None:
        """Print the circuit using the platform's draw function.

        Args:
            x (Optional[np.ndarray], optional): input example data for printing the circuit. Defaults to None.

        Raises:
            NotImplementedError: not supported platform
        """
        if self.platform == "pennylane":
            self._print_pennylane_circuit(x)
        else:
            raise NotImplementedError(f"Printing circuit is not supported in {self.platform}.")

    def _print_pennylane_circuit(self, x: Optional[np.ndarray] = None) -> None:
        """Print the circuit using PennyLane's draw function.
        if x is None, random input data is used for printing the circuit.

        Args:
            x (Optional[np.ndarray], optional): input example data for printing the circuit. Defaults to None.
        """
        if x is None:
            x = np.random.rand(1, self.n_qubits)

        self.check_input_shape(x)
        print(qml.draw(self.feature_map)(x))
