import numpy as np
from qiskit import QuantumCircuit

from qxmt.feature_maps.qiskit.base import QiskitBaseFeatureMap


class RotationFeatureMap(QiskitBaseFeatureMap):
    """Multi-axis rotation feature map class."""

    def __init__(self, n_qubits: int, reps: int, rotation_axis: list[str]) -> None:
        """Initialize the multi-axis rotation feature map class.
        Args:
            n_qubits (int): number of qubits
            reps (int): number of repetitions
            rotation_axis (list[str]): list of rotation axis ("X", "Y", "Z")
        """
        super().__init__(n_qubits)
        self.reps: int = reps
        self.rotation_axis: list[str] = rotation_axis
        self._axis_to_method = {
            "X": "rx",
            "Y": "ry",
            "Z": "rz",
        }
        self._validate_rotation_axis()

    def _validate_rotation_axis(self) -> None:
        if not all(axis in self._axis_to_method for axis in self.rotation_axis):
            raise ValueError(
                f"Invalid rotation axis: {self.rotation_axis}. Valid axes are {list(self._axis_to_method.keys())}"
            )

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of rotation feature map.
        Args:
            x (np.ndarray): input data
        """
        self.circuit = QuantumCircuit(self.n_qubits)
        for _ in range(self.reps):
            for axis in self.rotation_axis:
                method = getattr(self.circuit, self._axis_to_method[axis])
                for i in range(self.n_qubits):
                    method(float(x[i]), i)


class HRotationFeatureMap(QiskitBaseFeatureMap):
    """Hadamard and multi-axis rotation feature map class."""

    def __init__(self, n_qubits: int, reps: int, rotation_axis: list[str]) -> None:
        """Initialize the hadamard and multi-axis rotation feature map class.
        Args:
            n_qubits (int): number of qubits
            reps (int): number of repetitions
            rotation_axis (list[str]): list of rotation axis ("X", "Y", "Z")
        """
        super().__init__(n_qubits)
        self.reps: int = reps
        self.rotation_axis: list[str] = rotation_axis
        self._axis_to_method = {
            "X": "rx",
            "Y": "ry",
            "Z": "rz",
        }
        self._validate_rotation_axis()

    def _validate_rotation_axis(self) -> None:
        if not all(axis in self._axis_to_method for axis in self.rotation_axis):
            raise ValueError(
                f"Invalid rotation axis: {self.rotation_axis}. Valid axes are {list(self._axis_to_method.keys())}"
            )

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of hadamard and multi-axis rotation feature map.
        Args:
            x (np.ndarray): input data
        """
        self.circuit = QuantumCircuit(self.n_qubits)
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                self.circuit.h(i)
            for axis in self.rotation_axis:
                method = getattr(self.circuit, self._axis_to_method[axis])
                for i in range(self.n_qubits):
                    method(float(x[i]), i)
