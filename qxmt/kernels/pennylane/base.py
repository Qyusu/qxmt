from abc import abstractmethod
from typing import Any, Callable

import numpy as np
import pennylane as qml
from pennylane.measurements import SampleMP, StateMP
from rich.progress import track

from qxmt.devices.base import BaseDevice
from qxmt.exceptions import ModelSettingError
from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.kernels.base import STATE_VECTOR_BLOCK_SIZE, BaseKernel


class PennyLaneBaseKernel(BaseKernel):
    """PennyLane base kernel class.
    This class is the base class for all PennyLane kernels.
    It provides the basic functionality for all PennyLane kernels.
    """

    def __init__(self, device: BaseDevice, feature_map: BaseFeatureMap | Callable[[np.ndarray], None]) -> None:
        super().__init__(device, feature_map)
        self.state_memory: dict[tuple[float, ...], float | np.ndarray] = {}
        self._initialize_qnode()

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

    def _initialize_qnode(self) -> None:
        if self.feature_map is None:
            raise ModelSettingError("Feature map must be provided for PennyLaneBaseKernel.")

        if self.is_sampling:
            self.qnode = qml.QNode(
                self._circuit_for_sampling, device=self.device.get_device(), cache="auto", diff_method=None
            )
        else:
            self.qnode = qml.QNode(
                self._circuit_for_state_vector, device=self.device.get_device(), cache="auto", diff_method=None
            )

    def get_circuit_spec(self, x: np.ndarray) -> dict[str, Any]:
        return qml.specs(self.qnode)(x)  # type: ignore

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
