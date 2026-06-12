from logging import Logger
from typing import Any, Optional, cast

import numpy as np
from qiskit import QuantumCircuit

from qxmt.constants import QISKIT_PLATFORM
from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


class QiskitBaseFeatureMap(BaseFeatureMap):
    """Qiskit base feature map class."""

    def __init__(self, n_qubits: int) -> None:
        super().__init__(QISKIT_PLATFORM, n_qubits)
        self.circuit: QuantumCircuit

    def draw(
        self,
        x: Optional[np.ndarray] = None,
        x_dim: Optional[int] = None,
        format: str = "text",
        logger: Logger = LOGGER,
        **kwargs: Any,
    ) -> None:
        """Draw the circuit using Qiskit's draw function.
        Args:
            x (Optional[np.ndarray], optional): input example data for drawing the circuit. Defaults to None.
            x_dim (Optional[int], optional): dimension of input data. Defaults to None.
            format (str, optional): format of the drawing the circuit. Select "text", "mpl", or "latex". Defaults to "text".
            logger (Logger, optional): logger object. Defaults to LOGGER.
        Raises:
            ValueError: if both 'x' and 'x_dim' are None, or if 'format' is invalid.
        """
        if (x is None) and (x_dim is None):
            raise ValueError("Either 'x' or 'x_dim' argument must be provided.")

        x_sample = x[0] if x is not None else np.random.rand(1, cast(int, x_dim))[0]
        self.feature_map(x_sample)
        logger.info(self.circuit.draw(output=format, **kwargs))
