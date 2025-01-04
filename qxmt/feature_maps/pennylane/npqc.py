import numpy as np
import pennylane as qml

from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class NPQCFeatureMap(BaseFeatureMap):
    """NPQC feature map class.
    Reference: https://arxiv.org/abs/2108.01039

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> import numpy as np
        >>> from qxmt.feature_maps.pennylane.npqc import NPQCFeatureMap
        >>> feature_map = NPQCFeatureMap(2, 2, 0.1)
        >>> feature_map.draw(x_dim=2)
        0: ──RY(1.60)──RZ(1.59)──RY(1.57)─╭●──RY(1.60)──RZ(1.59)──RY(1.57)─╭●──RY(1.60)─┤
        1: ──RY(1.60)──RZ(1.59)───────────╰Z───────────────────────────────╰Z───────────┤
    """

    def __init__(self, n_qubits: int, reps: int, c: float) -> None:
        """ "Initialize the NPQC feature map class.

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
        self._validation()

    def _validation(self) -> None:
        """Validate the NPQC feature map."""
        if self.n_qubits % 2 != 0:
            raise ValueError(f"NPQC feature map requires an even number of qubits. but got {self.n_qubits}")

    def _calculate_target_wire(self, idx: int, r_idx: int, n_qubits: int) -> int:
        """Calculate target wire for controlled-Z gate.

        Args:
            idx (int): source wire index
            r_idx (int): repetition index
            n_qubits (int): number of qubits

        Returns:
            int: target wire index
        """
        remaining_divisions = r_idx + 1
        target_offset = 0
        while remaining_divisions % 2 == 0:
            remaining_divisions //= 2
            target_offset += 1
        return (idx + target_offset * 2 + 1) % n_qubits

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of NPQC feature map.

        Args:
            x (np.ndarray): input data
        """
        data_idx = 0
        # Apply RY and RZ rotations based on input
        for i in range(self.n_qubits):
            qml.RY(self.c * x[data_idx % len(x)] + np.pi / 2, wires=i)
            data_idx += 1
            qml.RZ(self.c * x[data_idx % len(x)] + np.pi / 2, wires=i)
            data_idx += 1

        for r_idx in range(self.reps):
            for i in range(0, self.n_qubits - 1, 2):
                qml.RY(qml.numpy.array(np.pi / 2), wires=i)

                # Calculate and apply controlled-Z gate
                target_wire = self._calculate_target_wire(i, r_idx, self.n_qubits)
                qml.CZ(wires=[i, target_wire])

                # Add RY and optional RZ gates for parameterized input
                qml.RY(self.c * x[data_idx % len(x)] + np.pi / 2, wires=i)
                data_idx += 1
                if r_idx + 1 < self.reps:
                    qml.RZ(self.c * x[data_idx % len(x)] + np.pi / 2, wires=i)
                    data_idx += 1
