from typing import Callable, cast

import numpy as np
import pennylane as qml
from pennylane.measurements.probs import ProbabilityMP
from pennylane.measurements.sample import SampleMP

from qxmt.devices.base import BaseDevice
from qxmt.exceptions import ModelSettingError
from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.kernels.base import BaseKernel
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

    def _circuit(self, x1: np.ndarray, x2: np.ndarray) -> ProbabilityMP | SampleMP:
        if self.feature_map is None:
            raise ModelSettingError("Feature map must be provided for FidelityKernel.")

        self.feature_map(x1)
        qml.adjoint(self.feature_map)(x2)  # type: ignore

        if self.is_sampling:
            return qml.sample(op=qml.PauliZ(wires=self.n_qubits))
        else:
            return qml.probs(wires=range(self.n_qubits))

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute the fidelity kernel value between two data points.

        Args:
            x1 (np.ndarray): numpy array representing the first data point
            x2 (np.ndarray): numpy array representing the second data point

        Returns:
            tuple[float, np.ndarray]: fidelity kernel value and probability distribution
        """
        # qnode = qml.QNode(self._circuit, device=self.device(), cache=False)  # type: ignore
        qnode = qml.QNode(self._circuit, device=self.device.get_device(), cache=False)  # type: ignore
        result = qnode(x1, x2)

        if self.is_sampling:
            # convert the sample results to probability distribution
            # shots must be over 0 when sampling mode
            probs = sample_results_to_probs(result, self.n_qubits, cast(int, self.device.shots))
        else:
            # use theoretical probability distribution
            probs = np.array(result)

        kernel_value = probs[0]  # get |0..0> state probability

        return kernel_value, probs
