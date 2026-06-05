from typing import Any

from qxmt.configs import DeviceConfig
from qxmt.constants import PENNYLANE_PLATFORM, QULACS_PLATFORM
from qxmt.devices.amazon import AMAZON_BRAKET_DEVICES
from qxmt.devices.base import BaseDevice
from qxmt.devices.ibmq import IBMQ_REAL_DEVICES
from qxmt.exceptions import AmazonBraketSettingError, IBMQSettingError, InvalidPlatformError
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)

PENNYLANE_DEFAULT_DEVICE_NAME: str = "default.qubit"
QULACS_DEFAULT_DEVICE_NAME: str = "cpu.simulator"


class DeviceBuilder:
    """
    Builder class for quantum devices. This class abstracts the instantiation of quantum device objects
    across multiple platforms (e.g., PennyLane, IBMQ, Amazon Braket) based on a unified configuration.

    This builder enables users to create a device instance by simply providing a DeviceConfig, without
    worrying about the underlying platform-specific details.

    Args:
        config (DeviceConfig): Configuration for the quantum device. This includes platform, device name,
            backend name, number of qubits, shots, and other specific options.
        logger (Any): Logger instance for logging.

    Methods:
        build() -> BaseDevice:
            Instantiates and returns a quantum device object corresponding to the specified platform and
            device name.

    Returns:
        BaseDevice: An instance of a quantum device suitable for the specified platform and configuration.

    Example:
        >>> from qxmt.configs import DeviceConfig
        >>> from qxmt.devices.builder import DeviceBuilder
        >>> config = DeviceConfig(
        ...     platform="pennylane",
        ...     device_name="default.qubit",
        ...     backend_name=None,
        ...     n_qubits=2,
        ...     shots=1000,
        ...     device_options={"seed": 42},
        ... )
        >>> device = DeviceBuilder(config).build()
    """

    def __init__(
        self,
        config: DeviceConfig,
        logger: Any = LOGGER,
    ) -> None:
        """Initialize the device builder.

        Args:
            config (DeviceConfig): Configuration for the quantum device. It is element of the ExperimentConfig.
        """
        self.config: DeviceConfig = config
        self.logger: Any = logger

    def build(self) -> BaseDevice:
        """Build a quantum device. it can be a general-purpose device overseeing multiple platforms.

        Returns:
            BaseDevice: General-purpose device overseeing multiple platforms
        """
        platform = self.config.platform
        device_name = self.config.device_name
        backend_name = self.config.backend_name
        n_qubits = self.config.n_qubits
        shots = self.config.shots
        device_options = self.config.device_options

        if platform == PENNYLANE_PLATFORM:
            if device_name in IBMQ_REAL_DEVICES:
                try:
                    from qxmt.devices.ibmq_device import IBMQDevice
                except ImportError as exc:
                    raise IBMQSettingError(
                        "IBMQ support requires optional dependencies. "
                        'Install them with `pip install "qxmt[pennylane-qiskit]"`.'
                    ) from exc

                return IBMQDevice(
                    platform=platform,
                    device_name=device_name,
                    backend_name=backend_name,
                    n_qubits=n_qubits,
                    shots=shots,
                    device_options=device_options,
                    logger=self.logger,
                )
            elif device_name in AMAZON_BRAKET_DEVICES:
                try:
                    from qxmt.devices.amazon_device import AmazonBraketDevice
                except ImportError as exc:
                    raise AmazonBraketSettingError(
                        "Amazon Braket support requires optional dependencies. "
                        'Install them with `pip install "qxmt[amazon-braket]"`.'
                    ) from exc

                return AmazonBraketDevice(
                    platform=platform,
                    device_name=device_name,
                    backend_name=backend_name,
                    n_qubits=n_qubits,
                    shots=shots,
                    device_options=device_options,
                    logger=self.logger,
                )
            else:
                from qxmt.devices.pennylane_device import PennyLaneDevice

                return PennyLaneDevice(
                    platform=platform,
                    device_name=device_name if device_name is not None else PENNYLANE_DEFAULT_DEVICE_NAME,
                    backend_name=backend_name,
                    n_qubits=n_qubits,
                    shots=shots,
                    device_options=device_options,
                    logger=self.logger,
                )
        elif platform == QULACS_PLATFORM:
            from qxmt.devices.qulacs_device import QulacsDevice

            return QulacsDevice(
                platform=platform,
                device_name=device_name if device_name is not None else QULACS_DEFAULT_DEVICE_NAME,
                backend_name=backend_name,
                n_qubits=n_qubits,
                shots=shots,
                device_options=device_options,
                logger=self.logger,
            )
        else:
            raise InvalidPlatformError(f'"{platform}" is not implemented.')
