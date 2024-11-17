import numpy as np
import pennylane as qml

from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class YZCXFeatureMap(BaseFeatureMap):
    """YZCX feature map class.
    Reference: https://arxiv.org/abs/2108.01039

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> import numpy as np
        >>> from qxmt.feature_maps.pennylane.yzcx import YZCXFeatureMap
        >>> feature_map = YZCXFeatureMap(2, 2, 0.1, 0)
        >>> feature_map.draw(x_dim=2)
        0: ──RY(0.03)──RY(3.42)──RZ(0.05)──RZ(5.88)─╭●──RY(0.03)──RY(5.39)──RZ(0.05)──RZ(0.21)
        1: ─────────────────────────────────────────╰X──RY(0.03)──RY(5.13)──RZ(0.05)──RZ(0.02)
        ──────────────────────────────────────────┤
        ──RY(0.03)───RY(4.58)──RZ(0.05)──RZ(1.10)─┤
    """

    def __init__(self, n_qubits: int, reps: int, c: float, seed: int) -> None:
        """ "Initialize the YZCX feature map class.

        Args:
            n_qubits (int): number of qubits
            reps (int): number of repetitions
            c (float): scaling factor
            seed (int): random seed
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits: int = n_qubits
        self.reps: int = reps
        self.c: float = c
        self.seed: int = seed

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of YZCX feature map.

        Args:
            x (np.ndarray): input data
        """
        rng = np.random.default_rng(self.seed)
        data_idx = 0
        for r_idx in range(self.reps):
            for i in range(self.n_qubits):
                # Apply rotaion Y gate by data value and random angle
                qml.RY(self.c * x[data_idx % len(x)], wires=i)
                ry_angle = 2.0 * np.pi * rng.random()
                qml.RY(ry_angle, wires=i)
                data_idx += 1

                # Apply rotaion Z gate by data value and random angle
                qml.RZ(self.c * x[data_idx % len(x)], wires=i)
                rz_angle = 2.0 * np.pi * rng.random()
                qml.RZ(rz_angle, wires=i)
                data_idx += 1

                # Apply CNOT gate based on the current repetition and qubit index
                if (i % 2 == r_idx % 2) and (i + 1 < self.n_qubits):
                    qml.CNOT(wires=[i, i + 1])
