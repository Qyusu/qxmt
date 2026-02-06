from logging import Logger
from typing import Any, Optional, cast

import numpy as np
import pennylane as qml

from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class PennyLaneBaseFeatureMap(BaseFeatureMap):
    """PennyLane base feature map class.

    Args:
        BaseFeatureMap (_type_): base feature map class
    """

    def __init__(self, n_qubits: int) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)

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

        match format:
            case "default":
                logger.info(qml.draw(qnode=self.feature_map, **kwargs)(x_sample))
            case "mpl":
                logger.info(qml.draw_mpl(qnode=self.feature_map, **kwargs)(x_sample))
            case _:
                raise ValueError(f"Invalid format '{format}' for drawing the circuit")
