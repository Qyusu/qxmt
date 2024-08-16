from typing import Callable

import numpy as np
import pennylane as qml


def print_pennylane_circuit(feature_map: Callable, x: np.ndarray) -> None:
    """Print the circuit using PennyLane's draw function.

    Args:
        feature_map (Callable): feature map function that defines the circuit
        x (np.ndarray): input example data for printing the circuit
    """
    print(qml.draw(feature_map)(x))
