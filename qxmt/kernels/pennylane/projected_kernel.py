from typing import Callable, Literal

import numpy as np
import pennylane as qml
from pennylane.measurements.sample import SampleMP
from pennylane.measurements.state import StateMP

from qxmt.devices import BaseDevice
from qxmt.feature_maps import BaseFeatureMap
from qxmt.kernels.pennylane.base import PennyLaneBaseKernel


class ProjectedKernel(PennyLaneBaseKernel):
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

    def _circuit(self, x: np.ndarray) -> None:
        self.feature_map(x)

        # apply projection operators for calculating the expected values by Z basis
        if self.projection == "x":
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
        elif self.projection == "y":
            for i in range(self.n_qubits):
                qml.RY(qml.numpy.array(np.pi / 2), wires=i)

    def _circuit_for_sampling(self, *args: np.ndarray) -> SampleMP | list[SampleMP]:
        self._validate_circuit_args(args, 1, "ProjectedKernel._circuit_for_sampling")
        x = args[0]
        self._circuit(x)

        return self._get_sampling_measurement()

    def _circuit_for_state_vector(self, *args: np.ndarray) -> StateMP:
        self._validate_circuit_args(args, 1, "ProjectedKernel._circuit_for_state_vector")
        x = args[0]
        self._circuit(x)

        return qml.state()

    def _process_state_vector(self, state_vector: np.ndarray) -> np.ndarray:
        """Process the raw state vector for projected kernel computation.

        Args:
            state_vector (np.ndarray): Raw state vector from quantum circuit

        Returns:
            np.ndarray: Expected values calculated from the probability distribution
        """
        probs = np.abs(state_vector) ** 2
        return self._calculate_expected_values(probs)

    def _compute_kernel_block(self, block1: np.ndarray, block2: np.ndarray) -> np.ndarray:
        """Compute projected kernel values for blocks of expected values.

        Args:
            block1 (np.ndarray): First block of expected values
            block2 (np.ndarray): Second block of expected values

        Returns:
            np.ndarray: Computed RBF kernel block
        """
        # compute the squared Euclidean distance between blocks
        # ||x - y||² = ||x||² + ||y||² - 2⟨x, y⟩
        a_norm2 = np.sum(block1**2, axis=1).reshape(-1, 1)
        b_norm2 = np.sum(block2**2, axis=1).reshape(1, -1)
        cross_term = np.dot(block1, block2.T)
        sq_dist = a_norm2 + b_norm2 - 2 * cross_term

        # RBF kernel
        kernel_block = np.exp(-self.gamma * sq_dist)
        return kernel_block

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
        if not self.is_sampling:
            raise ValueError("_compute_by_sampling method is only available in sampling mode.")

        x1_result = self.qnode(x1)
        x2_result = self.qnode(x2)

        x1_probs = self._convert_sampling_results_to_probs(x1_result)
        x2_probs = self._convert_sampling_results_to_probs(x2_result)

        # compute expected values for projection operators
        x1_projected = self._calculate_expected_values(x1_probs)
        x2_projected = self._calculate_expected_values(x2_probs)

        # compute gaussian kernel value based on the projected measurement results
        kernel_value = np.exp(-self.gamma * np.sum((x1_projected - x2_projected) ** 2))

        return kernel_value, (x1_probs + x2_probs) / 2
