from collections import Counter
from pathlib import Path
from typing import Callable, Optional, cast

import numpy as np
import pennylane as qml
from pennylane.measurements.probs import ProbabilityMP
from pennylane.measurements.sample import SampleMP

from qxmt.devices.base import BaseDevice
from qxmt.exceptions import ModelSettingError
from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.kernels.base import BaseKernel


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
        self.qnode = qml.QNode(self._circuit, self.device())

    def _circuit(self, x1: np.ndarray, x2: np.ndarray) -> ProbabilityMP | SampleMP:
        if self.feature_map is None:
            raise ModelSettingError("Feature map must be provided for FidelityKernel.")

        self.feature_map(x1)
        qml.adjoint(self.feature_map)(x2)  # type: ignore

        if self.is_sampling:
            return qml.sample(wires=range(self.n_qubits))
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
        result = self.qnode(x1, x2)

        if self.is_sampling:
            # validate sampleing results for getting the each state probability
            self._validate_sampling_values(result)

            # convert the sample results to bit strings
            # ex) shots=3, n_qubits=2, [[0, 0], [1, 1], [0, 0]] => ["00", "11", "00"]
            result = [result] if np.array(result).ndim == 1 else result
            bit_strings = ["".join(map(str, sample)) for sample in result]
            all_states = self._generate_all_observable_states(state_pattern="01")

            # count the number of each state
            count_dict = Counter(bit_strings)
            state_counts = [count_dict.get(state, 0) for state in all_states]

            # convert the count to the probability
            probs = np.array(state_counts) / cast(int, self.device.shots)  # shots must be over 0
        else:
            # use theoretical probability distribution
            probs = result
            if isinstance(probs, qml.operation.Tensor):
                probs = probs.numpy()

        kernel_value = probs[0]  # get |0..0> state probability

        return kernel_value, probs
