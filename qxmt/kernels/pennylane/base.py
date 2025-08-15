from abc import abstractmethod
from typing import Callable, cast

import numpy as np
import pennylane as qml
from pennylane.measurements import SampleMP, StateMP
from rich.progress import track

from qxmt.devices.base import BaseDevice
from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.kernels.base import STATE_VECTOR_BLOCK_SIZE, BaseKernel
from qxmt.kernels.sampling import sample_results_to_probs


class PennyLaneBaseKernel(BaseKernel):
    """PennyLane base kernel class.
    This class is the base class for all PennyLane kernels.
    It provides the basic functionality for all PennyLane kernels.
    """

    def __init__(self, device: BaseDevice, feature_map: BaseFeatureMap | Callable[[np.ndarray], None]) -> None:
        super().__init__(device, feature_map)
        self._qnode: qml.QNode | None = None
        self.state_memory: dict[tuple[float, ...], float | np.ndarray] = {}

    @abstractmethod
    def _circuit_for_sampling(self, *args: np.ndarray) -> SampleMP | list[SampleMP]:
        """Circuit for sampling mode.

        Args:
            *args: Variable number of numpy arrays. Can be:
                - Single array (x) for single input circuits
                - Two arrays (x1, x2) for fidelity-based circuits

        Returns:
            SampleMP | list[SampleMP]: Measurement result
        """
        pass

    @abstractmethod
    def _circuit_for_state_vector(self, *args: np.ndarray) -> StateMP:
        """Circuit for state vector mode.

        Args:
            *args: Variable number of numpy arrays. Can be:
                - Single array (x) for single input circuits
                - Two arrays (x1, x2) for fidelity-based circuits

        Returns:
            StateMP: State measurement result
        """
        pass

    @property
    def qnode(self) -> qml.QNode:
        if self._qnode is None:
            if self.is_sampling:
                self._qnode = qml.QNode(
                    self._circuit_for_sampling, device=self.device.get_device(), cache="auto", diff_method=None
                )
            else:
                self._qnode = qml.QNode(
                    self._circuit_for_state_vector, device=self.device.get_device(), cache="auto", diff_method=None
                )
        return self._qnode

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
                x_state = self.qnode(np.array(x_tuple))
                self.state_memory[x_tuple] = self._process_state_vector(x_state)

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

    def _get_sampling_measurement(self) -> SampleMP | list[SampleMP]:
        """Get appropriate sampling measurement based on device type.

        Returns:
            SampleMP | list[SampleMP]: Measurement instruction
        """
        if self.device.is_amazon_device():
            # Amazon Braket does not support directly sample by computational basis
            return [qml.sample(op=qml.PauliZ(wires=i)) for i in range(self.n_qubits)]
        else:
            return qml.sample(wires=range(self.n_qubits))

    def _convert_sampling_results_to_probs(self, result: list | np.ndarray) -> np.ndarray:
        """Convert sampling results to probability distribution.

        Args:
            result: Raw sampling results from quantum circuit

        Returns:
            np.ndarray: Probability distribution
        """
        if self.device.is_amazon_device():
            # PauliZ basis convert to computational basis (-1->1, 1->0)
            binary_result = (np.array(result).T == -1).astype(int)
            # convert the sample results to probability distribution
            # shots must be over 0 when sampling mode
            probs = sample_results_to_probs(binary_result, self.n_qubits, cast(int, self.device.shots))
        else:
            result_array = np.array(result) if isinstance(result, list) else result
            # convert the sample results to probability distribution
            # shots must be over 0 when sampling mode
            probs = sample_results_to_probs(result_array, self.n_qubits, cast(int, self.device.shots))

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
