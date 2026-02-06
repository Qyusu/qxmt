import numpy as np
from qulacs import QuantumCircuit

from qxmt.feature_maps.qulacs.base import QulacsBaseFeatureMap


class RotationFeatureMap(QulacsBaseFeatureMap):
    """Multi-axis rotation feature map class.

    Args:
        QulacsBaseFeatureMap (_type_): base feature map class for qulacs
    """

    def __init__(self, n_qubits: int, reps: int, rotation_axis: list[str]) -> None:
        super().__init__(n_qubits)
        self.reps: int = reps
        self.rotation_axis: list[str] = rotation_axis
        self._axis_to_adder = {
            "X": "add_RX_gate",
            "Y": "add_RY_gate",
            "Z": "add_RZ_gate",
        }

        self._validate_rotation_axis()

    def _validate_rotation_axis(self) -> None:
        if not all(axis in self._axis_to_adder for axis in self.rotation_axis):
            raise ValueError(
                f"Invalid rotation axis: {self.rotation_axis}. Valid axes are {list(self._axis_to_adder.keys())}"
            )

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of rotation feature map.

        Args:
            x (np.ndarray): input data
        """
        self.circuit: QuantumCircuit = QuantumCircuit(self.n_qubits)
        for _ in range(self.reps):
            for axis in self.rotation_axis:
                adder = getattr(self.circuit, self._axis_to_adder[axis])

                for i in range(self.n_qubits):
                    adder(i, float(x[i]))


class HRotationFeatureMap(QulacsBaseFeatureMap):
    """Hadamard and multi-axis rotation feature map class.

    Args:
        QulacsBaseFeatureMap (_type_): base feature map class for qulacs
    """

    def __init__(self, n_qubits: int, reps: int, rotation_axis: list[str]) -> None:
        """Initialize the Hadamard and multi axis rotation feature map class.

        Args:
            n_qubits (int): number of qubits
            reps (int): number of repetitions
            rotation_axis (list[str]): list of rotation axis
        """
        super().__init__(n_qubits)
        self.reps: int = reps
        self.rotation_axis: list[str] = rotation_axis
        self._axis_to_adder = {
            "X": "add_RX_gate",
            "Y": "add_RY_gate",
            "Z": "add_RZ_gate",
        }

        self._validate_rotation_axis()

    def _validate_rotation_axis(self) -> None:
        if not all(axis in self._axis_to_adder for axis in self.rotation_axis):
            raise ValueError(
                f"Invalid rotation axis: {self.rotation_axis}. Valid axes are {list(self._axis_to_adder.keys())}"
            )

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of hadamard and multi-axis rotation feature map.

        Args:
            x (np.ndarray): input data
        """
        self.circuit: QuantumCircuit = QuantumCircuit(self.n_qubits)
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                self.circuit.add_H_gate(i)
            for axis in self.rotation_axis:
                adder = getattr(self.circuit, self._axis_to_adder[axis])

                for i in range(self.n_qubits):
                    adder(i, float(x[i]))
