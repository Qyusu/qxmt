from typing import Callable, cast

import numpy as np
import pennylane as qml
from pennylane.measurements.sample import SampleMP
from pennylane.measurements.state import StateMP
from rich.progress import track

from qxmt.devices import BaseDevice
from qxmt.exceptions import ModelSettingError
from qxmt.feature_maps import BaseFeatureMap
from qxmt.kernels.base import STATE_VECTOR_BLOCK_SIZE
from qxmt.kernels.pennylane import PennyLaneBaseKernel
from qxmt.kernels.sampling import sample_results_to_probs


class FidelityKernel(PennyLaneBaseKernel):
    """Fidelity kernel class.
    The fidelity kernel is a quantum kernel that computes the kernel value based on the fidelity
    between two quantum states.

    Args:
        BaseKernel (_type_): base class of kernel

    Examples:
        >>> import numpy as np
        >>> from qxmt.kernels.pennylane.fidelity_kernel import FidelityKernel
        >>> from qxmt.feature_maps.pennylane.defaults import ZZFeatureMap
        >>> from qxmt.configs import DeviceConfig
        >>> from qxmt.devices.builder import DeviceBuilder
        >>> config = DeviceConfig(
        ...     platform="pennylane",
        ...     name="default.qubit",
        ...     n_qubits=2,
        ...     shots=1024,
        >>> )
        >>> device = DeviceBuilder(config).build()
        >>> feature_map = ZZFeatureMap(2, 2)
        >>> kernel = FidelityKernel(device, feature_map)
        >>> x1 = np.random.rand(2)
        >>> x2 = np.random.rand(2)
        >>> kernel.compute(x1, x2)
        0.14
    """

    def __init__(
        self,
        device: BaseDevice,
        feature_map: BaseFeatureMap | Callable[[np.ndarray], None],
    ) -> None:
        """Initialize the FidelityKernel class.

        Args:
            device (BaseDevice): device instance for quantum computation
            feature_map (BaseFeatureMap | Callable[[np.ndarray], None]): feature map instance or function
        """
        super().__init__(device, feature_map)

    def _circuit_for_sampling(self, *args: np.ndarray) -> SampleMP | list[SampleMP]:
        if len(args) != 2:
            raise ValueError("FidelityKernel._circuit_for_sampling requires exactly 2 arguments (x1, x2)")

        x1, x2 = args
        self.feature_map(x1)
        qml.adjoint(self.feature_map)(x2)  # type: ignore

        if (self.is_sampling) and (self.device.is_amazon_device()):
            # Amazon Braket does not support directry sample by computational basis
            return [qml.sample(op=qml.PauliZ(wires=i)) for i in range(self.n_qubits)]
        else:
            return qml.sample(wires=range(self.n_qubits))

    def _circuit_for_state_vector(self, *args: np.ndarray) -> StateMP:
        if len(args) != 1:
            raise ValueError("FidelityKernel._circuit_for_state_vector requires exactly 1 argument (x)")

        x = args[0]
        self.feature_map(x)

        return qml.state()

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
        self._initialize_qnode()
        if self.qnode is None:
            raise RuntimeError("QNode is not initialized.")

        unique_inputs = set([tuple(x) for x in x1_array] + [tuple(x) for x in x2_array])
        if show_progress:
            bar_label = f" ({bar_label})" if bar_label else ""
            iterator = track(unique_inputs, description=f"Computing Kernel Matrix{bar_label}")
        else:
            iterator = unique_inputs

        # compute the state vector for each data point
        for x_tuple in iterator:
            if x_tuple not in self.state_memory:
                self.state_memory[x_tuple] = self.qnode(np.array(x_tuple))

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

                inner_block = np.dot(block1, np.conj(block2.T))
                kernel_block = np.abs(inner_block) ** 2
                kernel_matrix[i_start:i_end, j_start:j_end] = kernel_block

        return kernel_matrix

    def _compute_by_sampling(self, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute the fidelity kernel value between two data points.
        This method is only available in the sampling mode.
        Each kernel value computed by sampling the quantum circuit.

        Args:
            x1 (np.ndarray): numpy array representing the first data point
            x2 (np.ndarray): numpy array representing the second data point

        Returns:
            tuple[float, np.ndarray]: fidelity kernel value and probability distribution
        """
        if not self.is_sampling:
            raise ValueError("_compute_by_sampling method is only available in sampling mode.")

        self._initialize_qnode()
        if self.qnode is None:
            raise RuntimeError("QNode is not initialized.")

        if (self.is_sampling) and (self.device.is_amazon_device()):
            result = self.qnode(x1, x2)
            # PauliZ basis convert to computational basis (-1->1, 1->0)
            binary_result = (np.array(result).T == -1).astype(int)
            # convert the sample results to probability distribution
            # shots must be over 0 when sampling mode
            probs = sample_results_to_probs(binary_result, self.n_qubits, cast(int, self.device.shots))
        else:
            result = self.qnode(x1, x2)
            # convert the sample results to probability distribution
            # shots must be over 0 when sampling mode
            probs = sample_results_to_probs(result, self.n_qubits, cast(int, self.device.shots))

        kernel_value = probs[0]  # get |0..0> state probability

        return kernel_value, probs
