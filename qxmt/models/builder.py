from qxmt.exceptions import (
    InvalidConfigError,
    InvalidFeatureMapError,
    InvalidKernelError,
    InvalidModelNameError,
    InvalidPlatformError,
)
from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.kernels.base import BaseKernel
from qxmt.models.base import BaseMLModel
from qxmt.models.qsvm import QSVM
from qxmt.models.schema import DeviceConfig, ModelConfig
from qxmt.utils.yaml import load_class_from_yaml


class ModelBuilder:
    def __init__(
        self,
        raw_config: dict,
    ) -> None:
        self.raw_config: dict = raw_config
        self.device_config: DeviceConfig = self._get_device_config()
        self.model_config: ModelConfig = self._get_model_config()
        self.feature_map: BaseFeatureMap
        self.kernel: BaseKernel
        self.model: BaseMLModel

    def _get_device_config(self, key: str = "device") -> DeviceConfig:
        """Get quantum device configurations.

        Args:
            key (str, optional): key for device configuration. Defaults to "device".

        Raises:
            InvalidConfigError: key is not in the configuration file.
        """
        if key not in self.raw_config:
            raise InvalidConfigError(f"Key '{key}' is not in the configuration file.")

        return DeviceConfig(**self.raw_config[key])

    def _get_model_config(self, key: str = "model") -> ModelConfig:
        """Get quantum model configurations.

        Args:
            key (str, optional): key for model configuration. Defaults to "model".

        Raises:
            InvalidConfigError: key is not in the configuration file.
        """
        if key not in self.raw_config:
            raise InvalidConfigError(f"Key '{key}' is not in the configuration file.")

        model_config_dict = self.raw_config["model"]
        model_config_dict["feature_map"] = self.raw_config.get("feature_map", None)
        model_config_dict["kernel"] = self.raw_config.get("kernel", None)

        return ModelConfig(**model_config_dict)

    def _set_device(self) -> None:
        """Set quantum device.

        Raises:
            InvalidPlatformError: platform is not implemented.
        """
        if self.device_config.platform == "pennylane":
            from pennylane import qml

            self.device = qml.device(
                name=self.device_config.name,
                wires=self.device_config.n_qubits,
                shots=self.device_config.shots,
            )
        else:
            raise InvalidPlatformError(f'"{self.device_config.platform}" is not implemented.')

    def _set_feature_map(self) -> None:
        """Set quantum feature map."""
        feature_map_config = self.model_config.feature_map
        if feature_map_config is None:
            return
        else:
            self.feature_map = load_class_from_yaml(
                feature_map_config.model_dump(), dynamic_params={"n_qubits": self.device_config.n_qubits}
            )

    def _set_kernel(self) -> None:
        """Set quantum kernel."""
        kernel = self.model_config.kernel
        if kernel is None:
            return
        # [TODO]: switch by platform
        elif kernel.name == "fidelity":
            from qxmt.kernels.pennylane.fidelity_kernel import FidelityKernel

            self.kernel = FidelityKernel(self.device, self.feature_map)
        else:
            raise InvalidKernelError(f'"{kernel.name}" is not implemented.')

    def _set_model(self) -> None:
        """Set quantum model."""
        if self.model_config.name == "qsvm":
            self.model = QSVM(kernel=self.kernel, **self.model_config.params)
        else:
            raise InvalidModelNameError(f'"{self.model_config.name}" is not implemented.')

    def build(self) -> BaseMLModel:
        """Build quantum model."""
        self._set_device()
        self._set_feature_map()
        self._set_kernel()
        self._set_model()

        return self.model
