from qxmt.configs import DeviceConfig
from qxmt.devices.base import BaseDevice


class DeviceBuilder:
    def __init__(
        self,
        config: DeviceConfig,
    ) -> None:
        self.config: DeviceConfig = config

    def build(self) -> BaseDevice:
        """Build a quantum device. it can be a general-purpose device overseeing multiple platforms.

        Returns:
            BaseDevice: General-purpose device overseeing multiple platforms
        """
        return BaseDevice(
            platform=self.config.platform,
            name=self.config.name,
            n_qubits=self.config.n_qubits,
            shots=self.config.shots,
        )
