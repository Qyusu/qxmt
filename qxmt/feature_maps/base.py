from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Callable, Optional, cast

import numpy as np

from qxmt.constants import SUPPORTED_PLATFORMS
from qxmt.exceptions import InputShapeError, InvalidPlatformError
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class BaseFeatureMap(ABC):
    """Base class for feature map.
    This class is used to define the feature map for quantum machine learning.
    The feature map defined a quantum circuit that maps input data to quantum states.
    Provide a common interface within the QXMT library by absorbing differences between multiple platforms.
    User can define their own feature map by inheriting this class and defined in the 'feature_map' method.

    Examples:
        >>> import numpy as np
        >>> import pennylane as qml
        >>> from qxmt.feature_maps.base import BaseFeatureMap
        >>> class CustomFeatureMap(BaseFeatureMap):
        ...     def __init__(self, platform: str, n_qubits: int) -> None:
        ...         super().__init__(platform, n_qubits)
        ...
        ...     def feature_map(self, x: np.ndarray) -> None:
        ...         # define quantum circuit
        ...         qml.RX(x[0, 0], wires=0)
        ...
        >>> feature_map = CustomFeatureMap("pennylane", 2)
        >>> feature_map(np.random.rand(1, 2))
    """

    def __init__(self, platform: str, n_qubits: int) -> None:
        """Initialize the feature map class.
        [NOTE]: currently, only supports 'pennylane' platform. multi-platform support will be added in the future.

        Args:
            platform (str): name of quantum platform
            n_qubits (int): number of qubits
        """
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

    def check_input_dim_eq_nqubits(self, x: np.ndarray, idx: int = -1) -> None:
        """Check if the input data dimension matches the number of qubits.

        Args:
            x (np.ndarray): input data
            idx (int, optional): index of the dimension of qubit. Defaults to -1.

        Raises:
            InputShapeError: input data dimension does not match the number of qubits.
        """
        if x.shape[idx] != self.n_qubits:
            raise InputShapeError("Input data shape does not match the number of qubits.")

    def draw(
        self,
        x: Optional[np.ndarray] = None,
        x_dim: Optional[int] = None,
        format: str = "default",
        logger: Logger = LOGGER,
        **kwargs: Any,
    ) -> None:
        """Draw the circuit using the platform's draw function.

        Args:
            x (Optional[np.ndarray], optional): input example data for drawing the circuit. Defaults to None.
            x_dim (Optional[int], optional): dimension of input data. Defaults to None.
            format (str, optional): format of the drawing the circuit. Select "defalt" or "mpl". Defaults to "default".
            logger (Logger, optional): logger object. Defaults to LOGGER.

        Raises:
            NotImplementedError: not supported platform
        """
        if (x is None) and (x_dim is None):
            raise ValueError("Either 'x' or 'x_dim' argument must be provided.")

        x_sample = x[0] if x is not None else np.random.rand(1, cast(int, x_dim))[0]

        if self.platform == "pennylane":
            import pennylane as qml

            match format:
                case "default":
                    logger.info(qml.draw(qnode=self.feature_map, **kwargs)(x_sample))
                case "mpl":
                    logger.info(qml.draw_mpl(qnode=self.feature_map, **kwargs)(x_sample))
                case _:
                    raise ValueError(f"Invalid format '{format}' for drawing the circuit")
        else:
            raise NotImplementedError(f'"draw" method is not supported in {self.platform}.')


class FeatureMapFromFunc(BaseFeatureMap):
    """Wrap the feature map function to the BaseFeatureMap class."""

    def __init__(self, platform: str, n_qubits: int, feature_map_func: Callable[[np.ndarray], None]) -> None:
        super().__init__(platform, n_qubits)
        self.feature_map_func = feature_map_func

    def feature_map(self, x: np.ndarray) -> None:
        self.feature_map_func(x)
