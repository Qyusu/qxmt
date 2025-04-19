from typing import Callable, cast

import numpy as np
import pennylane as qml
from pennylane.measurements.sample import SampleMP
from pennylane.measurements.state import StateMP
from rich.progress import track

from qxmt.devices import BaseDevice
from qxmt.exceptions import ModelSettingError
from qxmt.feature_maps import BaseFeatureMap
from qxmt.kernels import BaseKernel
from qxmt.kernels.sampling import sample_results_to_probs


class FidelityKernel(BaseKernel):
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
        self.qnode = None
        self.state_memory_1 = {}
        self.state_memory_2 = {}

    def _initialize_qnode(self) -> None:
        if (self.qnode is None) and (self.is_sampling):
            self.qnode = qml.QNode(self._circuit, device=self.device.get_device(), cache=False, diff_method=None)
        elif (self.qnode is None) and (not self.is_sampling):
            self.qnode = qml.QNode(
                self._circuit_state_vector, device=self.device.get_device(), cache=False, diff_method=None
            )

    def _circuit(self, x1: np.ndarray, x2: np.ndarray) -> SampleMP | list[SampleMP]:
        if self.feature_map is None:
            raise ModelSettingError("Feature map must be provided for FidelityKernel.")

        self.feature_map(x1)
        qml.adjoint(self.feature_map)(x2)  # type: ignore

        if (self.is_sampling) and (self.device.is_amazon_device()):
            # Amazon Braket does not support directry sample by computational basis
            return [qml.sample(op=qml.PauliZ(wires=i)) for i in range(self.n_qubits)]
        else:
            return qml.sample(wires=range(self.n_qubits))

    def _circuit_state_vector(self, x: np.ndarray) -> StateMP:
        if self.feature_map is None:
            raise ModelSettingError("Feature map must be provided for FidelityKernel.")

        self.feature_map(x)

        return qml.state()

    def _compute_matrix_by_state_vector(
        self, x1_array: np.ndarray, x2_array: np.ndarray, bar_label: str = "", show_progress: bool = True
    ) -> np.ndarray:
        """Compute the kernel matrix based on the state vector.
        This method is only available in the non-sampling mode.
        Each kernel value computed by theoritically probability distribution by state vector.

        Args:
            x1_array (np.ndarray): numpy array representing the all data points (ex: Train data)
            x2_array (np.ndarray): numpy array representing the all data points (ex: Train data, Test data)
            bar_label (str): label for progress bar
            show_progress (bool): flag for showing progress bar

        Returns:
            np.ndarray: computed kernel matrix
        """
        self._initialize_qnode()
        if self.qnode is None:
            raise RuntimeError("QNode is not initialized.")

        if len(x1_array) > len(x2_array):
            x1_array, x2_array = x2_array, x1_array

        state_memory_1 = {}
        state_memory_2 = {}
        bar_label = f" ({bar_label})" if bar_label else ""
        for i, x in enumerate(x1_array):
            if i not in state_memory_1:
                state_memory_1[i] = self.qnode(x)
                if np.array_equal(x, x2_array[i]):
                    state_memory_2[i] = state_memory_1[i]

        kernel_matrix = np.zeros((len(x1_array), len(x2_array)))

        if show_progress:
            iterator = track(range(len(x1_array)), description=f"Computing Kernel Matrix{bar_label}")
        else:
            iterator = range(len(x1_array))

        for i in iterator:
            for j, x2 in enumerate(x2_array):
                if j not in state_memory_2:
                    state_memory_2[j] = self.qnode(x2)
                kernel_matrix[i, j] = np.abs(np.dot(np.conj(state_memory_1[i]), state_memory_2[j])) ** 2

        if len(x1_array) > len(x2_array):
            kernel_matrix = kernel_matrix.T

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
