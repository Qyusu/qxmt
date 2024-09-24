from typing import Callable

import numpy as np
import pennylane as qml
from pennylane.measurements.probs import ProbabilityMP

from qxmt.devices.base import BaseDevice
from qxmt.exceptions import ModelSettingError
from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.kernels.base import BaseKernel


class FidelityKernel(BaseKernel):
    """Fidelity kernel class.

    Args:
        BaseKernel (_type_): base class of kernel

    Examples:
        >>> import numpy as np
        >>> from typing import Callable
        >>> from qxmt.kernels.pennylane.fidelity_kernel import FidelityKernel
        >>> from qxmt.feature_maps.pennylane.defaults import ZZFeatureMap
        >>> from qxmt.configs import DeviceConfig
        >>> from qxmt.devices.builder import DeviceBuilder
        >>> config = DeviceConfig(
        ...     platform="pennylane",
        ...     name="default.qubit",
        ...     n_qubits=2,
        ...     shots=1000,
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

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute the fidelity kernel value between two data points.

        Args:
            x1 (np.ndarray): numpy array representing the first data point
            x2 (np.ndarray): numpy array representing the second data point

        Returns:
            float: fidelity kernel value
        """

        def circuit(x1: np.ndarray, x2: np.ndarray) -> ProbabilityMP:
            if self.feature_map is None:
                raise ModelSettingError("Feature map must be provided for FidelityKernel.")

            self.feature_map(x1)
            qml.adjoint(self.feature_map)(x2)  # type: ignore

            return qml.probs(wires=range(self.n_qubits))

        qnode = qml.QNode(circuit, self.device())
        probs = qnode(x1, x2)
        if isinstance(probs, qml.operation.Tensor):
            probs = probs.numpy()

        kernel_value = probs[0]  # get |00> state probability

        return kernel_value
