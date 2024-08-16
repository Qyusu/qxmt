from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional

import numpy as np

from qxmt.constants import SUPPORTED_PLATFORMS
from qxmt.exceptions import InputShapeError, InvalidPlatformError
from qxmt.feature_maps.pennylane.utils import output_pennylane_circuit
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


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

    def output_circuit(self, x: Optional[np.ndarray] = None, logger: Logger = LOGGER) -> None:
        """Output the circuit using the platform's draw function.

        Args:
            x (Optional[np.ndarray], optional): input example data for output the circuit. Defaults to None.
            logger (Logger, optional): logger object. Defaults to LOGGER.

        Raises:
            NotImplementedError: not supported platform
        """
        if x is None:
            x = np.random.rand(1, self.n_qubits)
        self.check_input_shape(x)

        if self.platform == "pennylane":
            output_pennylane_circuit(self.feature_map, x, logger)
        else:
            raise NotImplementedError(f'"output_circuit" method is not supported in {self.platform}.')
