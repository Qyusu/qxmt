from typing import Any, Optional

from qxmt.exceptions import InvalidPlatformError


class BaseDevice:
    def __init__(self, platform: str, name: str, n_qubits: int, shots: Optional[int]) -> None:
        self.platform = platform
        self.name = name
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
        if self.platform == "pennylane":
            from pennylane import qml

            self.device = qml.device(
                name=self.name,
                wires=self.n_qubits,
                shots=self.shots,
            )
        else:
            raise InvalidPlatformError(f'"{self.platform}" is not implemented.')
