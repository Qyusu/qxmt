from collections.abc import Callable

import numpy as np
from qiskit import QuantumCircuit

from qxmt.feature_maps.qiskit.base import QiskitBaseFeatureMap


class XXFeatureMap(QiskitBaseFeatureMap):
    """XX feature map class."""

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
                self.circuit.rx(float(x[i]), i)
            for i in range(0, self.n_qubits - 1):
                angle = 2 * (np.pi - x[i]) * (np.pi - x[i + 1])
                self.circuit.rxx(float(angle), i, i + 1)


class YYFeatureMap(QiskitBaseFeatureMap):
    """YY feature map class."""

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
                self.circuit.h(i)
                self.circuit.ry(float(x[i]), i)
            for i in range(0, self.n_qubits - 1):
                angle = 2 * (np.pi - x[i]) * (np.pi - x[i + 1])
                self.circuit.ryy(float(angle), i, i + 1)


class ZZFeatureMap(QiskitBaseFeatureMap):
    """ZZ feature map class.

    Uses Qiskit's built-in ``ZZFeatureMap`` when available and falls back to an
    equivalent nearest-neighbor implementation otherwise.
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
        circuit = self._build_with_qiskit_zz_feature_map(x)
        if circuit is None:
            self.circuit = self._build_manual_circuit(x)
        else:
            self.circuit = circuit

    def _build_with_qiskit_zz_feature_map(self, x: np.ndarray) -> QuantumCircuit | None:
        try:
            from qiskit.circuit.library import ZZFeatureMap as QiskitZZFeatureMap
        except ImportError:
            return None

        def data_map_func(values: np.ndarray) -> float:
            if len(values) == 1:
                return float(values[0])
            return float((np.pi - values[0]) * (np.pi - values[1]))

        try:
            circuit = QiskitZZFeatureMap(
                feature_dimension=self.n_qubits,
                reps=self.reps,
                entanglement="linear",
                data_map_func=cast_data_map_func(data_map_func),
            )
            return circuit.assign_parameters({param: float(value) for param, value in zip(circuit.parameters, x)})
        except TypeError:
            return None

    def _build_manual_circuit(self, x: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.n_qubits)
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                circuit.h(i)
                circuit.rz(float(x[i]), i)
            for i in range(0, self.n_qubits - 1):
                angle = 2 * (np.pi - x[i]) * (np.pi - x[i + 1])
                circuit.rzz(float(angle), i, i + 1)
        return circuit


def cast_data_map_func(func: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
    """Cast a data map function to the expected type for Qiskit's ZZFeatureMap."""
    return func
