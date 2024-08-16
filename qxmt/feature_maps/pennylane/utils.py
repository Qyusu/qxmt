from logging import Logger
from typing import Callable

import numpy as np
import pennylane as qml

from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


def output_pennylane_circuit(feature_map: Callable, x: np.ndarray, logger: Logger = LOGGER) -> None:
    """Output the circuit using PennyLane's draw function.

    Args:
        feature_map (Callable): feature map function that defines the circuit
        x (np.ndarray): input example data for output the circuit
        logger (Logger, optional): logger object. Defaults to LOGGER.
    """
    logger.info(qml.draw(feature_map)(x))
