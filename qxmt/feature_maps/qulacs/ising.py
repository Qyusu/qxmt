import numpy as np
from qulacs import QuantumCircuit

from qxmt.feature_maps.qulacs.base import QulacsBaseFeatureMap


class XXFeatureMap(QulacsBaseFeatureMap):
    """XX feature map class.

    Args:
        QulacsBaseFeatureMap (_type_): base feature map class for Qulacs
    """

    def __init__(self, n_qubits: int, reps: int) -> None:
        """Initialize the XX feature map class.

        Args:
            n_qubits (int): number of qubits
            reps (int): number of repetitions
        """
        super().__init__(n_qubits)
        self.reps: int = reps

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of XX feature map.

        Args:
            x (np.ndarray): input data
        """
        self.circuit = QuantumCircuit(self.n_qubits)
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                self.circuit.add_RX_gate(i, float(x[i]))
            for i in range(0, self.n_qubits - 1):
                # IsingXX(phi) = exp(-i * phi/2 * X@X)
                # target angle is 2 * (pi - x[i]) * (pi - x[i+1])
                angle = 2 * (np.pi - x[i]) * (np.pi - x[i + 1])
                self.circuit.add_multi_Pauli_rotation_gate([i, i + 1], [1, 1], float(angle))


class YYFeatureMap(QulacsBaseFeatureMap):
    """YY feature map class.

    Args:
        QulacsBaseFeatureMap (_type_): base feature map class for Qulacs
    """

    def __init__(self, n_qubits: int, reps: int) -> None:
        """Initialize the YY feature map class.

        Args:
            n_qubits (int): number of qubits
            reps (int): number of repetitions
        """
        super().__init__(n_qubits)
        self.reps: int = reps

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of YY feature map.

        Args:
            x (np.ndarray): input data
        """
        self.circuit = QuantumCircuit(self.n_qubits)
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                self.circuit.add_H_gate(i)
                self.circuit.add_RY_gate(i, float(x[i]))
            for i in range(0, self.n_qubits - 1):
                # IsingYY(phi) = exp(-i * phi/2 * Y@Y)
                angle = 2 * (np.pi - x[i]) * (np.pi - x[i + 1])
                self.circuit.add_multi_Pauli_rotation_gate([i, i + 1], [2, 2], float(angle))


class ZZFeatureMap(QulacsBaseFeatureMap):
    """ZZ feature map class.

    Args:
        QulacsBaseFeatureMap (_type_): base feature map class for Qulacs
    """

    def __init__(self, n_qubits: int, reps: int) -> None:
        """Initialize the ZZ feature map class.

        Args:
            n_qubits (int): number of qubits
            reps (int): number of repetitions
        """
        super().__init__(n_qubits)
        self.reps: int = reps

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of ZZ feature map.

        Args:
            x (np.ndarray): input data
        """
        self.circuit = QuantumCircuit(self.n_qubits)
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                self.circuit.add_H_gate(i)
                self.circuit.add_RZ_gate(i, float(x[i]))
            for i in range(0, self.n_qubits - 1):
                # IsingZZ(phi) = exp(-i * phi/2 * Z@Z)
                angle = 2 * (np.pi - x[i]) * (np.pi - x[i + 1])
                self.circuit.add_multi_Pauli_rotation_gate([i, i + 1], [3, 3], float(angle))
