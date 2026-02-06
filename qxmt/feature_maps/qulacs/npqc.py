import numpy as np
from qulacs import QuantumCircuit

from qxmt.feature_maps.qulacs.base import QulacsBaseFeatureMap


class NPQCFeatureMap(QulacsBaseFeatureMap):
    """NPQC feature map class.
    Reference: https://arxiv.org/abs/2108.01039

    Args:
        QulacsBaseFeatureMap (_type_): base feature map class for Qulacs
    """

    def __init__(self, n_qubits: int, reps: int, c: float) -> None:
        """ "Initialize the NPQC feature map class.

        Args:
            n_qubits (int): number of qubits
            reps (int): number of repetitions
            c (float): scaling factor
        """
        super().__init__(n_qubits)
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
        self.circuit = QuantumCircuit(self.n_qubits)
        data_idx = 0
        # Apply RY and RZ rotations based on input
        for i in range(self.n_qubits):
            self.circuit.add_RY_gate(i, float(self.c * x[data_idx % len(x)] + np.pi / 2))
            data_idx += 1
            self.circuit.add_RZ_gate(i, float(self.c * x[data_idx % len(x)] + np.pi / 2))
            data_idx += 1

        for r_idx in range(self.reps):
            for i in range(0, self.n_qubits - 1, 2):
                self.circuit.add_RY_gate(i, float(np.pi / 2))

                # Calculate and apply controlled-Z gate
                target_wire = self._calculate_target_wire(i, r_idx, self.n_qubits)
                self.circuit.add_CZ_gate(i, target_wire)

                # Add RY and optional RZ gates for parameterized input
                self.circuit.add_RY_gate(i, float(self.c * x[data_idx % len(x)] + np.pi / 2))
                data_idx += 1
                if r_idx + 1 < self.reps:
                    self.circuit.add_RZ_gate(i, float(self.c * x[data_idx % len(x)] + np.pi / 2))
                    data_idx += 1
