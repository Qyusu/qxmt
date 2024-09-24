from qxmt.configs import DeviceConfig
from qxmt.devices.base import BaseDevice


class DeviceBuilder:
    """
    Builder class for quantum devices.

    Examples:
        >>> from qxmt.configs import DeviceConfig
        >>> from qxmt.devices.builder import DeviceBuilder
        >>> config = DeviceConfig(
        ...     platform="pennylane",
        ...     name="default.qubit",
        ...     n_qubits=2,
        ...     shots=1000,
        ... )
        >>> device = DeviceBuilder(config).build()
    """

    def __init__(
        self,
        config: DeviceConfig,
    ) -> None:
        """Initialize the device builder.

        Args:
            config (DeviceConfig): Configuration for the quantum device. It is element of the ExperimentConfig.
        """
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
