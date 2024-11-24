from typing import Any, Optional

import numpy as np

from qxmt.exceptions import InvalidPlatformError


class BaseDevice:
    """General-purpose device class for experiment.
    This class is abstracted to oversee multiple platforms.
    Provide a common interface within the QXMT library by absorbing differences between platforms.

    Examples:
        >>> from qxmt.devices.base import BaseDevice
        >>> device = BaseDevice(platform="pennylane", name="default.qubit", n_qubits=2, shots=100)
    """

    def __init__(
        self, platform: str, name: str, n_qubits: int, shots: Optional[int], random_seed: Optional[int] = None
    ) -> None:
        """Initialize the quantum device.

        Args:
            platform (str): platform name (ex: pennylane, qulacs, etc.)
            name (str): device name provided by the platform (ex: default.qubit, default.tensor, etc.)
            n_qubits (int): number of qubits
            shots (Optional[int]): number of shots for the quantum circuit
            random_seed (Optional[int]): random seed for the quantum device
        """
        self.platform = platform
        self.name = name
        self.n_qubits = n_qubits
        self.shots = shots
        self.random_seed = random_seed
        self._set_device()

    def __call__(self) -> Any:
        return self.device

    def _set_device(self) -> None:
        """Set quantum device.

        Raises:
            InvalidPlatformError: platform is not implemented.
        """
        if self.platform == "pennylane":
            import pennylane as qml

            self.device = qml.device(
                name=self.name,
                wires=self.n_qubits,
                shots=self.shots,
                seed=np.random.default_rng(self.random_seed) if self.random_seed is not None else "global",
            )
        else:
            raise InvalidPlatformError(f'"{self.platform}" is not implemented.')
