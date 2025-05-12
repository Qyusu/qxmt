from typing import Callable, Literal, cast

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


class ProjectedKernel(BaseKernel):
    """Projected kernel class.
    The projected kernel is a quantum kernel that projects the quantum state to a specific basis
    and computes the kernel value based on the projected measurement results.
    Reference: https://www.nature.com/articles/s41467-021-22539-9

    Args:
        BaseKernel (_type_): base class of kernel

    Examples:
        >>> import numpy as np
        >>> from qxmt.kernels.pennylane.projected_kernel import ProjectedKernel
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
        >>> kernel = ProjectedKernel(device, feature_map)
        >>> x1 = np.random.rand(2)
        >>> x2 = np.random.rand(2)
        >>> kernel.compute(x1, x2)
        0.86
    """

    def __init__(
        self,
        device: BaseDevice,
        feature_map: BaseFeatureMap | Callable[[np.ndarray], None],
        gamma: float = 1.0,
        projection: Literal["x", "y", "z"] = "z",
    ) -> None:
        """Initialize the ProjectedKernel class.

        Args:
            device (BaseDevice): device instance for quantum computation
            feature_map (BaseFeatureMap | Callable[[np.ndarray], None]): feature map instance or function
            gamma (float): gamma parameter for kernel computation
            projection (str): projection method for kernel computation
        """
        if projection not in ["x", "y", "z"]:
            raise ValueError("Projection method must be 'x', 'y', or 'z'.")

        super().__init__(device, feature_map)
        self.n_qubits = device.n_qubits
        self.gamma = gamma
        self.projection = projection
        self.qnode = None
        self.state_memory = {}

    def _initialize_qnode(self) -> None:
        if self.qnode is None:
            self.qnode = qml.QNode(self._circuit, device=self.device.get_device(), cache=False)

    def _process_measurement_results(self, results: list[float]) -> np.ndarray:
        """Process the measurement results based on the projection method.

        Args:
            results (list[float]): projected measurement results for each qubit

        Returns:
            np.ndarray: processed measurement results
        """
        if self.projection == "xyz_sum":
            sum_results: list[float] = []
            idx_size = 3  # x, y, z
            for i in range(0, self.n_qubits, idx_size):
                sum_result: float = np.sum([result for result in results[i : i + idx_size]])
                sum_results.append(sum_result)
            projected_results = np.array(sum_results)
        else:
            projected_results = np.array(results)

        return projected_results

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
        # calculate the expected values by Z basis for each qubit
        # when the projection method is "x" or "y", apply Hadamard or RY gate in _circuit method
        projected_exp_value = np.array([self._calculate_expected_values_by_z(probs, i) for i in range(self.n_qubits)])

        return projected_exp_value

    def _circuit(self, x: np.ndarray) -> SampleMP | list[SampleMP] | StateMP:
        if self.feature_map is None:
            raise ModelSettingError("Feature map must be provided for FidelityKernel.")

        self.feature_map(x)

        # apply projection operators for calculating the expected values by Z basis
        if self.projection == "x":
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
        elif self.projection == "y":
            for i in range(self.n_qubits):
                qml.RY(qml.numpy.array(np.pi / 2), wires=i)

        if (self.is_sampling) and (self.device.is_amazon_device()):
            # Amazon Braket does not support directry sample by computational basis
            return [qml.sample(op=qml.PauliZ(wires=i)) for i in range(self.n_qubits)]
        elif self.is_sampling:
            return qml.sample(wires=self.n_qubits)
        else:
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

        unique_inputs = set([tuple(x) for x in x1_array] + [tuple(x) for x in x2_array])
        if show_progress:
            bar_label = f" ({bar_label})" if bar_label else ""
            iterator = track(unique_inputs, description=f"Computing Kernel Matrix{bar_label}")
        else:
            iterator = unique_inputs

        for x_tuple in iterator:
            if x_tuple not in self.state_memory:
                x_state = self.qnode(np.array(x_tuple))
                self.state_memory[x_tuple] = self._calculate_expected_values(np.abs(x_state) ** 2)

        kernel_matrix = np.zeros((len(x1_array), len(x2_array)))
        for i, x1 in enumerate(x1_array):
            for j, x2 in enumerate(x2_array):
                kernel_matrix[i, j] = np.exp(
                    -self.gamma * np.sum((self.state_memory[tuple(x1)] - self.state_memory[tuple(x2)]) ** 2)
                )

        return kernel_matrix

    def _compute_by_sampling(self, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute the projected kernel value between two data points.
        This method is only available in the sampling mode.
        Each kernel value computed by sampling the quantum circuit.

        Args:
            x1 (np.ndarray): numpy array representing the first data point
            x2 (np.ndarray): numpy array representing the second data point

        Returns:
            tuple[float, np.ndarray]: projected kernel value and probability distribution
        """
        self._initialize_qnode()
        if self.qnode is None:
            raise RuntimeError("QNode is not initialized.")

        x1_result = self.qnode(x1)
        x2_result = self.qnode(x2)

        if (self.is_sampling) and (self.device.is_amazon_device()):
            # PauliZ basis convert to computational basis (-1->1, 1->0)
            x1_binary_result = (np.array(x1_result).T == -1).astype(int)
            x2_binary_result = (np.array(x2_result).T == -1).astype(int)
            # convert the sample results to probability distribution
            # shots must be over 0 when sampling mode
            x1_probs = sample_results_to_probs(x1_binary_result, self.n_qubits, cast(int, self.device.shots))
            x2_probs = sample_results_to_probs(x2_binary_result, self.n_qubits, cast(int, self.device.shots))
        else:
            # convert the sample results to probability distribution
            # shots must be over 0 when sampling mode
            x1_probs = sample_results_to_probs(x1_result, self.n_qubits, cast(int, self.device.shots))
            x2_probs = sample_results_to_probs(x2_result, self.n_qubits, cast(int, self.device.shots))

        # compute expected values for projection operators
        x1_projected = self._calculate_expected_values(x1_probs)
        x2_projected = self._calculate_expected_values(x2_probs)

        # compute gaussian kernel value based on the projected measurement results
        kernel_value = np.exp(-self.gamma * np.sum((x1_projected - x2_projected) ** 2))

        return kernel_value, (x1_probs + x2_probs) / 2
