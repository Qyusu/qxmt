import numpy as np
from qiskit import QuantumCircuit

from qxmt.feature_maps.qiskit.base import QiskitBaseFeatureMap


class YZCXFeatureMap(QiskitBaseFeatureMap):
    """YZCX feature map class.

    Reference: https://arxiv.org/abs/2108.01039
    """

    def __init__(self, n_qubits: int, reps: int, c: float, seed: int) -> None:
        """Initialize the YZCX feature map class.
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
            x (np.ndarray): input data"""
        rng = np.random.default_rng(self.seed)
        self.circuit = QuantumCircuit(self.n_qubits)
        data_idx = 0
        for r_idx in range(self.reps):
            for i in range(self.n_qubits):
                self.circuit.ry(float(self.c * x[data_idx % len(x)]), i)
                ry_angle = 2.0 * np.pi * rng.random()
                self.circuit.ry(float(ry_angle), i)
                data_idx += 1

                self.circuit.rz(float(self.c * x[data_idx % len(x)]), i)
                rz_angle = 2.0 * np.pi * rng.random()
                self.circuit.rz(float(rz_angle), i)
                data_idx += 1

                if (i % 2 == r_idx % 2) and (i + 1 < self.n_qubits):
                    self.circuit.cx(i, i + 1)
