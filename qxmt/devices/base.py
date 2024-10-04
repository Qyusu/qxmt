from typing import Any, Optional

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
        self, platform: str, name: Optional[str], method: Optional[str], n_qubits: int, shots: Optional[int]
    ) -> None:
        """Initialize the quantum device.

        Args:
            platform (str): platform name (ex: pennylane, qulacs, etc.)
            name (Optional[str]): device name provided by the platform (ex: default.qubit, default.tensor, etc.)
            method (Optional[str]): method for the quantum simulator (ex: statevector, density_matrix, etc.)
            n_qubits (int): number of qubits
            shots (Optional[int]): number of shots for the quantum circuit
        """
        self.platform = platform
        self.name = name
        self.method = method
        self.n_qubits = n_qubits
        self.shots = shots
        self._set_device()

    def __call__(self) -> Any:
        return self.device

    def _set_device(self) -> None:
        """Set quantum device.

        Raises:
            InvalidPlatformError: platform is not implemented.
        """
        match self.platform.lower():
            case "pennylane":
                from pennylane import qml

                self.device = qml.device(
                    name=self.name,
                    wires=self.n_qubits,
                    shots=self.shots,
                )
            case "qiskit":
                from qiskit_aer import AerSimulator

                self.device = AerSimulator(
                    method=self.method,
                    shots=self.shots,
                )
            case "qulacs":
                raise InvalidPlatformError(f'"{self.platform}" is not implemented. It is supported in the future.')
            case "cirq":
                raise InvalidPlatformError(f'"{self.platform}" is not implemented. It is supported in the future.')
            case _:
                raise InvalidPlatformError(f'"{self.platform}" is not implemented.')
