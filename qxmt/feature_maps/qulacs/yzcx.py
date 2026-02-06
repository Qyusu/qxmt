import numpy as np
from qulacs import QuantumCircuit

from qxmt.feature_maps.qulacs.base import QulacsBaseFeatureMap


class YZCXFeatureMap(QulacsBaseFeatureMap):
    """YZCX feature map class.
    Reference: https://arxiv.org/abs/2108.01039

    Args:
        QulacsBaseFeatureMap (_type_): base feature map class for Qulacs
    """

    def __init__(self, n_qubits: int, reps: int, c: float, seed: int) -> None:
        """ "Initialize the YZCX feature map class.

        Args:
            n_qubits (int): number of qubits
            reps (int): number of repetitions
            c (float): scaling factor
            seed (int): random seed
        """
        super().__init__(n_qubits)
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
        self.circuit = QuantumCircuit(self.n_qubits)
        data_idx = 0
        for r_idx in range(self.reps):
            for i in range(self.n_qubits):
                # Apply rotaion Y gate by data value and random angle
                self.circuit.add_RY_gate(i, float(self.c * x[data_idx % len(x)]))
                ry_angle = 2.0 * np.pi * rng.random()
                self.circuit.add_RY_gate(i, float(ry_angle))
                data_idx += 1

                # Apply rotaion Z gate by data value and random angle
                self.circuit.add_RZ_gate(i, float(self.c * x[data_idx % len(x)]))
                rz_angle = 2.0 * np.pi * rng.random()
                self.circuit.add_RZ_gate(i, float(rz_angle))
                data_idx += 1

                # Apply CNOT gate based on the current repetition and qubit index
                if (i % 2 == r_idx % 2) and (i + 1 < self.n_qubits):
                    self.circuit.add_CNOT_gate(i, i + 1)
