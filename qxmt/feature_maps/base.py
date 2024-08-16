from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from qxmt.constants import SUPPORTED_PLATFORMS
from qxmt.exceptions import InputShapeError, InvalidPlatformError
from qxmt.feature_maps.pennylane.utils import print_pennylane_circuit


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
        if x is None:
            x = np.random.rand(1, self.n_qubits)
        self.check_input_shape(x)

        if self.platform == "pennylane":
            print_pennylane_circuit(self.feature_map, x)
        else:
            raise NotImplementedError(f"Printing circuit is not supported in {self.platform}.")
