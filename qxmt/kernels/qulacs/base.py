from abc import abstractmethod
from typing import Callable, cast

import numpy as np
from qulacs import QuantumState
from rich.progress import track

from qxmt.devices import QulacsDevice
from qxmt.feature_maps import QulacsBaseFeatureMap
from qxmt.kernels.base import STATE_VECTOR_BLOCK_SIZE, BaseKernel


class QulacsBaseKernel(BaseKernel):
    """Qulacs base kernel class.
    This class is the base class for all Qulacs kernels.
    It provides the basic functionality for all Qulacs kernels.
    """

    def __init__(self, device: QulacsDevice, feature_map: QulacsBaseFeatureMap | Callable[[np.ndarray], None]) -> None:
        super().__init__(device, feature_map)
        self.state_memory: dict[tuple[float, ...], float | np.ndarray] = {}

    @abstractmethod
    def _circuit_for_sampling(self, *args: np.ndarray) -> np.ndarray:
        """Circuit for sampling mode.

        Args:
            *args: Variable number of numpy arrays. Can be:
                - Single array (x) for single input circuits
                - Two arrays (x1, x2) for fidelity-based circuits

        Returns:
            np.ndarray: Measurement result or probability distribution
        """
        pass

    @abstractmethod
    def _process_state_vector(self, state_vector: np.ndarray) -> np.ndarray:
        """Process the raw state vector into the format needed for kernel computation.

        Args:
            state_vector (np.ndarray): Raw state vector from quantum circuit

        Returns:
            np.ndarray: Processed state data for kernel computation
        """
        pass

    @abstractmethod
    def _compute_kernel_block(self, block1: np.ndarray, block2: np.ndarray) -> np.ndarray:
        """Compute kernel values for blocks of processed states.

        Args:
            block1 (np.ndarray): First block of processed states
            block2 (np.ndarray): Second block of processed states

        Returns:
            np.ndarray: Computed kernel block
        """
        pass

    def _compute_matrix_by_state_vector(
        self,
        x1_array: np.ndarray,
        x2_array: np.ndarray,
        bar_label: str = "",
        show_progress: bool = True,
        block_size: int = STATE_VECTOR_BLOCK_SIZE,
    ) -> np.ndarray:
        """Compute the kernel matrix based on the state vector.
        This method is only available in the non-sampling mode.
        Each kernel value computed by theoritically probability distribution by state vector.

        Args:
            x1_array (np.ndarray): numpy array representing the all data points (ex: Train data)
            x2_array (np.ndarray): numpy array representing the all data points (ex: Train data, Test data)
            bar_label (str): label for progress bar
            show_progress (bool): flag for showing progress bar
            block_size (int): block size for the batch computation

        Returns:
            np.ndarray: computed kernel matrix
        """
        unique_inputs = set([tuple(x) for x in x1_array] + [tuple(x) for x in x2_array])
        if show_progress:
            bar_label = f" ({bar_label})" if bar_label else ""
            iterator = track(unique_inputs, description=f"Computing Kernel Matrix{bar_label}")
        else:
            iterator = unique_inputs

        # compute the state vector for each data point
        for x_tuple in iterator:
            if x_tuple not in self.state_memory:
                # Create a new quantum state
                state = QuantumState(self.n_qubits)
                state.set_zero_state()

                # Apply feature map
                # cast to QulacsBaseFeatureMap to access circuit
                feature_map = cast(QulacsBaseFeatureMap, self.feature_map)
                feature_map.feature_map(np.array(x_tuple))

                # Update state with circuit
                feature_map.circuit.update_quantum_state(state)

                # Get vector and process it
                state_vec = state.get_vector()
                self.state_memory[x_tuple] = self._process_state_vector(state_vec)

        states1 = np.array([self.state_memory[tuple(x)] for x in x1_array])
        states2 = np.array([self.state_memory[tuple(x)] for x in x2_array])

        # batch compute the kernel matrix
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

    def _convert_sampling_results_to_probs(self, result: list | np.ndarray) -> np.ndarray:
        """Convert sampling results to probability distribution.

        Args:
            result: Raw sampling results from quantum circuit

        Returns:
            np.ndarray: Probability distribution
        """
        result_array = np.array(result) if isinstance(result, list) else result
        # convert the sample results to probability distribution
        # shots must be over 0 when sampling mode
        shots = cast(int, self.device.shots)
        n_states = 2**self.n_qubits

        # Qulacs sampling returns integers, so we can use bincount directly
        counts = np.bincount(result_array, minlength=n_states)

        # Ensure we only take the first n_states elements (though minlength handles this usually,
        # legitimate samples shouldn't exceed this unless qulacs is broken)
        if len(counts) > n_states:
            counts = counts[:n_states]

        probs = counts / shots

        return probs

    def _validate_circuit_args(self, args: tuple[np.ndarray, ...], expected_count: int, method_name: str) -> None:
        """Validate the number of arguments for circuit methods.

        Args:
            args: Arguments passed to circuit method
            expected_count: Expected number of arguments
            method_name: Name of the calling method for error message

        Raises:
            ValueError: If argument count doesn't match expected
        """
        if len(args) != expected_count:
            if expected_count == 1:
                raise ValueError(f"{method_name} requires exactly 1 argument (x)")
            elif expected_count == 2:
                raise ValueError(f"{method_name} requires exactly 2 arguments (x1, x2)")
            else:
                raise ValueError(f"{method_name} requires exactly {expected_count} arguments")
