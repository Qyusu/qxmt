from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from qiskit import QuantumCircuit

from qxmt.feature_maps.qiskit.base import QiskitBaseFeatureMap
from qxmt.kernels.qiskit.base import QiskitBaseKernel

if TYPE_CHECKING:
    from qxmt.devices.qiskit_device import QiskitDevice


class ProjectedKernel(QiskitBaseKernel):
    """Projected kernel class for Qiskit.
    The projected kernel is a quantum kernel that projects the quantum state to a specific basis
    and computes the kernel value based on the projected measurement results.
    Reference: https://www.nature.com/articles/s41467-021-22539-9

    Args:
        device (QiskitDevice): device instance for quantum computation
        feature_map (QiskitBaseFeatureMap | Callable[[np.ndarray], None]): feature map instance or function
        gamma (float): gamma parameter for kernel computation
        projection (str): projection method for kernel computation ("x", "y", "z")
    """

    def __init__(
        self,
        device: "QiskitDevice",
        feature_map: QiskitBaseFeatureMap | Callable[[np.ndarray], None],
        gamma: float = 1.0,
        projection: Literal["x", "y", "z"] = "z",
    ) -> None:
        if projection not in ["x", "y", "z"]:
            raise ValueError("Projection method must be 'x', 'y', or 'z'.")

        super().__init__(device, feature_map)
        self.gamma = gamma
        self.projection = projection

    def _apply_projection_gates(self, circuit: QuantumCircuit) -> None:
        """Apply basis rotation gates based on projection method."""
        if self.projection == "x":
            for i in range(self.n_qubits):
                circuit.h(i)
        elif self.projection == "y":
            for i in range(self.n_qubits):
                circuit.ry(float(np.pi / 2), i)

    def _build_state_vector_circuit(self, x: np.ndarray) -> QuantumCircuit:
        circuit = self._build_feature_map_circuit(x)
        self._apply_projection_gates(circuit)
        return circuit

    def _circuit_for_sampling(self, *args: np.ndarray) -> QuantumCircuit:
        """Build a projected circuit for sampling mode."""
        self._validate_circuit_args(args, 1, "ProjectedKernel._circuit_for_sampling")
        x = args[0]

        circuit = self._build_state_vector_circuit(x)
        return circuit

    def _calculate_expected_values_by_z(self, probs: np.ndarray, target_qubit: int) -> float:
        mask = 1 << target_qubit
        expval_z = 0.0
        for i, prob in enumerate(probs):
            if (i & mask) == 0:
                expval_z += prob
            else:
                expval_z -= prob
        return expval_z

    def _calculate_expected_values(self, probs: np.ndarray) -> np.ndarray:
        return np.array([self._calculate_expected_values_by_z(probs, i) for i in range(self.n_qubits)])

    def _process_state_vector(self, state_vector: np.ndarray) -> np.ndarray:
        probs = np.abs(state_vector) ** 2
        return self._calculate_expected_values(probs)

    def _compute_kernel_block(self, block1: np.ndarray, block2: np.ndarray) -> np.ndarray:
        a_norm2 = np.sum(block1**2, axis=1).reshape(-1, 1)
        b_norm2 = np.sum(block2**2, axis=1).reshape(1, -1)
        cross_term = np.dot(block1, block2.T)
        sq_dist = a_norm2 + b_norm2 - 2 * cross_term

        return np.exp(-self.gamma * sq_dist)

    def _compute_by_sampling(self, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute the projected kernel value by sampling."""
        if not self.is_sampling:
            raise ValueError("_compute_by_sampling method is only available in sampling mode.")

        x1_counts = self._run_sampling_circuit(self._circuit_for_sampling(x1))
        x2_counts = self._run_sampling_circuit(self._circuit_for_sampling(x2))

        x1_probs = self._convert_counts_to_probs(x1_counts)
        x2_probs = self._convert_counts_to_probs(x2_counts)

        x1_projected = self._calculate_expected_values(x1_probs)
        x2_projected = self._calculate_expected_values(x2_probs)
        kernel_value = float(np.exp(-self.gamma * np.sum((x1_projected - x2_projected) ** 2)))

        return kernel_value, (x1_probs + x2_probs) / 2
