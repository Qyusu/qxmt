from abc import abstractmethod
from typing import TYPE_CHECKING, Callable, cast

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from rich.progress import track

from qxmt.feature_maps.qiskit.base import QiskitBaseFeatureMap
from qxmt.kernels.base import STATE_VECTOR_BLOCK_SIZE, BaseKernel
from qxmt.kernels.sampling import generate_all_observable_states

if TYPE_CHECKING:
    from qxmt.devices.qiskit_device import QiskitDevice


class QiskitBaseKernel(BaseKernel):
    """Qiskit base kernel class."""

    def __init__(
        self, device: "QiskitDevice", feature_map: QiskitBaseFeatureMap | Callable[[np.ndarray], None]
    ) -> None:
        super().__init__(device, feature_map)
        self.state_memory: dict[tuple[float, ...], float | np.ndarray] = {}

    @abstractmethod
    def _circuit_for_sampling(self, *args: np.ndarray) -> QuantumCircuit:
        """Circuit for sampling mode."""
        pass

    @abstractmethod
    def _process_state_vector(self, state_vector: np.ndarray) -> np.ndarray:
        """Process the raw state vector into the format needed for kernel computation."""
        pass

    @abstractmethod
    def _compute_kernel_block(self, block1: np.ndarray, block2: np.ndarray) -> np.ndarray:
        """Compute kernel values for blocks of processed states."""
        pass

    def _build_feature_map_circuit(self, x: np.ndarray) -> QuantumCircuit:
        feature_map = cast(QiskitBaseFeatureMap, self.feature_map)
        feature_map.feature_map(x)
        return feature_map.circuit.copy()

    def _compute_matrix_by_state_vector(
        self,
        x1_array: np.ndarray,
        x2_array: np.ndarray,
        bar_label: str = "",
        show_progress: bool = True,
        block_size: int = STATE_VECTOR_BLOCK_SIZE,
    ) -> np.ndarray:
        """Compute the kernel matrix based on the state vector."""
        unique_inputs = set([tuple(x) for x in x1_array] + [tuple(x) for x in x2_array])
        if show_progress:
            bar_label = f" ({bar_label})" if bar_label else ""
            iterator = track(unique_inputs, description=f"Computing Kernel Matrix{bar_label}")
        else:
            iterator = unique_inputs

        for x_tuple in iterator:
            if x_tuple not in self.state_memory:
                circuit = self._build_state_vector_circuit(np.array(x_tuple))
                state_vec = Statevector.from_instruction(circuit).data
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
                kernel_matrix[i_start:i_end, j_start:j_end] = self._compute_kernel_block(block1, block2)

        return kernel_matrix

    def _build_state_vector_circuit(self, x: np.ndarray) -> QuantumCircuit:
        return self._build_feature_map_circuit(x)

    def _run_sampling_circuit(self, circuit: QuantumCircuit) -> dict[str, int]:
        if not self.is_sampling:
            raise ValueError("_run_sampling_circuit method is only available in sampling mode.")

        measured_circuit = circuit.copy()
        measured_circuit.measure_all()
        result = self.device.get_device().run(measured_circuit, shots=cast(int, self.device.shots)).result()
        return dict(result.get_counts())

    def _convert_counts_to_probs(self, counts: dict[str, int]) -> np.ndarray:
        shots = cast(int, self.device.shots)
        probs = np.zeros(2**self.n_qubits, dtype=np.float64)
        for bitstring, count in counts.items():
            normalized_bitstring = bitstring.replace(" ", "")
            probs[int(normalized_bitstring, 2)] += count / shots
        return probs

    def _validate_circuit_args(self, args: tuple[np.ndarray, ...], expected_count: int, method_name: str) -> None:
        if len(args) != expected_count:
            if expected_count == 1:
                raise ValueError(f"{method_name} requires exactly 1 argument (x)")
            elif expected_count == 2:
                raise ValueError(f"{method_name} requires exactly 2 arguments (x1, x2)")
            else:
                raise ValueError(f"{method_name} requires exactly {expected_count} arguments")

    def _all_observable_state_labels(self) -> list[str]:
        return generate_all_observable_states(self.n_qubits, state_pattern="01")
