from logging import Logger
from typing import Any, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
from qulacs import QuantumCircuit
from qulacsvis import circuit_drawer

from qxmt.constants import QULACS_PLATFORM
from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class QulacsBaseFeatureMap(BaseFeatureMap):
    """Qulacs base feature map class.

    Args:
        BaseFeatureMap (_type_): base feature map class
    """

    def __init__(self, n_qubits: int) -> None:
        super().__init__(QULACS_PLATFORM, n_qubits)
        self.circuit: QuantumCircuit

    def draw(
        self,
        x: Optional[np.ndarray] = None,
        x_dim: Optional[int] = None,
        format: str = "latex",
        logger: Logger = LOGGER,
        **kwargs: Any,
    ) -> None:
        """Draw the circuit using the platform's draw function.

        Args:
            x (Optional[np.ndarray], optional): input example data for drawing the circuit. Defaults to None.
            x_dim (Optional[int], optional): dimension of input data. Defaults to None.
            format (str, optional): format of the drawing the circuit. Select "latex", "latex_source", or "mpl". Defaults to "mpl".
            logger (Logger, optional): logger object. Defaults to LOGGER.

        Raises:
            NotImplementedError: not supported platform
        """
        if (x is None) and (x_dim is None):
            raise ValueError("Either 'x' or 'x_dim' argument must be provided.")

        x_sample = x[0] if x is not None else np.random.rand(1, cast(int, x_dim))[0]
        self.feature_map(x_sample)
        if format in ["latex", "latex_source"]:
            logger.info(circuit_drawer(self.circuit, format))
        elif format == "mpl":
            circuit_drawer(self.circuit, format)
            plt.show()
        else:
            raise ValueError(f"Invalid format '{format}' for drawing the circuit")
