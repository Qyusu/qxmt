from typing import Callable, Literal, cast

import numpy as np
from qulacs import QuantumState, gate
from rich.progress import track

from qxmt.devices import QulacsDevice
from qxmt.feature_maps import QulacsBaseFeatureMap
from qxmt.kernels.base import STATE_VECTOR_BLOCK_SIZE
from qxmt.kernels.qulacs.base import QulacsBaseKernel


class ProjectedKernel(QulacsBaseKernel):
    """Projected kernel class for Qulacs.
    The projected kernel is a quantum kernel that projects the quantum state to a specific basis
    and computes the kernel value based on the projected measurement results.
    Reference: https://www.nature.com/articles/s41467-021-22539-9

    Args:
        device (QulacsDevice): device instance for quantum computation
        feature_map (QulacsBaseFeatureMap | Callable[[np.ndarray], None]): feature map instance
        gamma (float): gamma parameter for kernel computation
        projection (str): projection method for kernel computation ("x", "y", "z")
    """

    def __init__(
        self,
        device: QulacsDevice,
        feature_map: QulacsBaseFeatureMap | Callable[[np.ndarray], None],
        gamma: float = 1.0,
        projection: Literal["x", "y", "z"] = "z",
    ) -> None:
        if projection not in ["x", "y", "z"]:
            raise ValueError("Projection method must be 'x', 'y', or 'z'.")

        super().__init__(device, feature_map)
        self.gamma = gamma
        self.projection = projection

    def _apply_projection_gates(self, state: QuantumState) -> None:
        """Apply basis rotation gates to the quantum state based on projection method."""
        if self.projection == "x":
            for i in range(self.n_qubits):
                gate.H(i).update_quantum_state(state)
        elif self.projection == "y":
            for i in range(self.n_qubits):
                gate.RY(i, np.pi / 2).update_quantum_state(state)

    def _circuit_for_sampling(self, *args: np.ndarray) -> np.ndarray:
        """Circuit for sampling mode."""
        self._validate_circuit_args(args, 1, "ProjectedKernel._circuit_for_sampling")
        x = args[0]

        state = QuantumState(self.n_qubits)
        state.set_zero_state()
        feature_map = cast(QulacsBaseFeatureMap, self.feature_map)
        feature_map.feature_map(x)
        feature_map.circuit.update_quantum_state(state)

        self._apply_projection_gates(state)

        return np.array(state.sampling(self.device.shots))

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
        projected_exp_value = np.array([self._calculate_expected_values_by_z(probs, i) for i in range(self.n_qubits)])
        return projected_exp_value

    def _process_state_vector(self, state_vector: np.ndarray) -> np.ndarray:
        probs = np.abs(state_vector) ** 2
        return self._calculate_expected_values(probs)

    def _compute_kernel_block(self, block1: np.ndarray, block2: np.ndarray) -> np.ndarray:
        # ||x - y||² = ||x||² + ||y||² - 2⟨x, y⟩
        a_norm2 = np.sum(block1**2, axis=1).reshape(-1, 1)
        b_norm2 = np.sum(block2**2, axis=1).reshape(1, -1)
        cross_term = np.dot(block1, block2.T)
        sq_dist = a_norm2 + b_norm2 - 2 * cross_term

        return np.exp(-self.gamma * sq_dist)

    def _compute_matrix_by_state_vector(
        self,
        x1_array: np.ndarray,
        x2_array: np.ndarray,
        bar_label: str = "",
        show_progress: bool = True,
        block_size: int = STATE_VECTOR_BLOCK_SIZE,
    ) -> np.ndarray:
        """Compute the kernel matrix based on the state vector.
        Overridden to include projection gates.
        """
        unique_inputs = set([tuple(x) for x in x1_array] + [tuple(x) for x in x2_array])
        if show_progress:
            bar_label = f" ({bar_label})" if bar_label else ""
            iterator = track(unique_inputs, description=f"Computing Kernel Matrix{bar_label}")
        else:
            iterator = unique_inputs

        for x_tuple in iterator:
            if x_tuple not in self.state_memory:
                state = QuantumState(self.n_qubits)
                state.set_zero_state()

                feature_map = cast(QulacsBaseFeatureMap, self.feature_map)
                feature_map.feature_map(np.array(x_tuple))
                feature_map.circuit.update_quantum_state(state)

                self._apply_projection_gates(state)

                state_vec = state.get_vector()
                self.state_memory[x_tuple] = self._process_state_vector(state_vec)

        states1 = np.array([self.state_memory[tuple(x)] for x in x1_array])
        states2 = np.array([self.state_memory[tuple(x)] for x in x2_array])

        n1 = len(states1)
        n2 = len(states2)
        kernel_matrix = np.zeros((n1, n2), dtype=np.float64)

        for i_start in range(0, n1, block_size):
            i_end = min(i_start + block_size, n1)
            block1 = states1[i_start:i_end]
            for j_start in range(0, n2, block_size):
                j_end = min(j_start + block_size, n2)
                block2 = states2[j_start:j_end]

                kernel_block = self._compute_kernel_block(block1, block2)
                kernel_matrix[i_start:i_end, j_start:j_end] = kernel_block

        return kernel_matrix

    def _compute_by_sampling(self, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute the projected kernel value by sampling."""
        if not self.is_sampling:
            raise ValueError("_compute_by_sampling method is only available in sampling mode.")

        x1_result = self._circuit_for_sampling(x1)
        x2_result = self._circuit_for_sampling(x2)

        x1_probs = self._convert_sampling_results_to_probs(x1_result)
        x2_probs = self._convert_sampling_results_to_probs(x2_result)

        # compute expected values for projection operators
        x1_projected = self._calculate_expected_values(x1_probs)
        x2_projected = self._calculate_expected_values(x2_probs)

        # compute gaussian kernel value based on the projected measurement results
        kernel_value = np.exp(-self.gamma * np.sum((x1_projected - x2_projected) ** 2))

        return kernel_value, (x1_probs + x2_probs) / 2
