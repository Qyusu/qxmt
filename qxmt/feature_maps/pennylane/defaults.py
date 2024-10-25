import numpy as np
import pennylane as qml

from qxmt.feature_maps.base import BaseFeatureMap

PENNYLANE_PLATFORM: str = "pennylane"


class RotationFeatureMap(BaseFeatureMap):
    """Multi-axis rotation feature map class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> import numpy as np
        >>> from qxmt.feature_maps.pennylane.defaults import RotationFeatureMap
        >>> feature_map = RotationFeatureMap(2, 2, ["X", "Y"])
        >>> feature_map(np.random.rand(1, 2))
    """

    def __init__(self, n_qubits: int, reps: int, rotation_axis: list[str]) -> None:
        """Initialize the multi axis rotation feature map class.

        Args:
            n_qubits (int): number of qubits
            reps (int): number of repetitions
            rotation_axis (list[str]): list of rotation axis
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.reps: int = reps
        self.rotation_axis: list[str] = rotation_axis

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of rotation feature map.

        Args:
            x (np.ndarray): input data
        """
        for _ in range(self.reps):
            for ax in self.rotation_axis:
                qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation=ax)


class HRotationFeatureMap(BaseFeatureMap):
    """Hadamard and multi-axis rotation feature map class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> import numpy as np
        >>> from qxmt.feature_maps.pennylane.defaults import HRotationFeatureMap
        >>> feature_map = HRotationFeatureMap(2, 2, ["X", "Y"])
        >>> feature_map(np.random.rand(1, 2))
    """

    def __init__(self, n_qubits: int, reps: int, rotation_axis: list[str]) -> None:
        """Initialize the Hadamard and multi axis rotation feature map class.

        Args:
            n_qubits (int): number of qubits
            reps (int): number of repetitions
            rotation_axis (list[str]): list of rotation axis
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.reps: int = reps
        self.rotation_axis: list[str] = rotation_axis

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of Hadamard and rotation feature map.

        Args:
            x (np.ndarray): input data
        """
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            for ax in self.rotation_axis:
                qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation=ax)


class XXFeatureMap(BaseFeatureMap):
    """XX feature map class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> import numpy as np
        >>> from qxmt.feature_maps.pennylane.defaults import XXFeatureMap
        >>> feature_map = XXFeatureMap(2, 2)
        >>> feature_map(np.random.rand(1, 2))
    """

    def __init__(self, n_qubits: int, reps: int) -> None:
        """Initialize the XX feature map class.

        Args:
            n_qubits (int): number of qubits
            reps (int): number of repetitions
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.reps: int = reps

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of XX feature map.

        Args:
            x (np.ndarray): input data
        """
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                qml.RX(x[i], wires=i)
            for i in range(0, self.n_qubits - 1):
                qml.IsingXX(2 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=[i, i + 1])


class YYFeatureMap(BaseFeatureMap):
    """YY feature map class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> import numpy as np
        >>> from qxmt.feature_maps.pennylane.defaults import YYFeatureMap
        >>> feature_map = YYFeatureMap(2, 2)
        >>> feature_map(np.random.rand(1, 2))
    """

    def __init__(self, n_qubits: int, reps: int) -> None:
        """Initialize the YY feature map class.

        Args:
            n_qubits (int): number of qubits
            reps (int): number of repetitions
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.reps: int = reps

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of YY feature map.

        Args:
            x (np.ndarray): input data
        """
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(x[i], wires=i)
            for i in range(0, self.n_qubits - 1):
                qml.IsingYY(2 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=[i, i + 1])


class ZZFeatureMap(BaseFeatureMap):
    """ZZ feature map class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> import numpy as np
        >>> from qxmt.feature_maps.pennylane.defaults import ZZFeatureMap
        >>> feature_map = ZZFeatureMap(2, 2)
        >>> feature_map(np.random.rand(1, 2))
    """

    def __init__(self, n_qubits: int, reps: int) -> None:
        """Initialize the ZZ feature map class.

        Args:
            n_qubits (int): number of qubits
            reps (int): number of repetitions
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.reps: int = reps

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of ZZ feature map.

        Args:
            x (np.ndarray): input data
        """
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x[i], wires=i)
            for i in range(0, self.n_qubits - 1):
                qml.IsingZZ(2 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=[i, i + 1])
