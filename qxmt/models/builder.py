from qxmt.exceptions import (
    InvalidFeatureMapError,
    InvalidKernelError,
    InvalidModelNameError,
    InvalidPlatformError,
)
from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.kernels.base import BaseKernel
from qxmt.models.base import BaseModel
from qxmt.models.qsvm import QSVM
from qxmt.models.schema import DeviceConfig, ModelConfig


class ModelBuilder:
    def __init__(
        self,
        device_config: dict,
        model_config: dict,
    ) -> None:
        self.device_config: DeviceConfig = DeviceConfig(**device_config)
        self.model_config: ModelConfig = ModelConfig(**model_config)
        self.feature_map: BaseFeatureMap
        self.kernel: BaseKernel
        self.model: BaseModel

    def _set_device(self) -> None:
        """Set quantum device.

        Raises:
            InvalidPlatformError: _description_
        """
        if self.device_config.platform == "pennylane":
            from pennylane import qml

            self.device = qml.device(self.device_config.name, wires=self.device_config.n_qubits)
        else:
            raise InvalidPlatformError(f'"{self.device_config.platform}" is not implemented.')

    def _set_feature_map(self) -> None:
        """Set quantum feature map."""
        feature_map = self.model_config.feature_map
        if feature_map is None:
            return
        # [TODO]: switch by platform
        elif feature_map == "ZZFeatureMap":
            from qxmt.feature_maps.pennylane.defaults import ZZFeatureMap

            self.feature_map = ZZFeatureMap(
                n_qubits=self.device_config.n_qubits,
                reps=self.model_config.map_params.get("reps", 1),
            )
        else:
            raise InvalidFeatureMapError(f'"{feature_map}" is not implemented.')

    def _set_kernel(self) -> None:
        """Set quantum kernel."""
        kernel_type = self.model_config.kernel
        if kernel_type is None:
            return
        # [TODO]: switch by platform
        elif kernel_type == "fidelity":
            from qxmt.kernels.pennylane.fidelity_kernel import FidelityKernel

            self.kernel = FidelityKernel(self.device, self.feature_map)
        else:
            raise InvalidKernelError(f'"{kernel_type}" is not implemented.')

    def _set_model(self) -> None:
        """Set quantum model."""
        if self.model_config.model_name == "qsvm":
            # self.model = QSVM(self.kernel, **self.model_config.model_params)
            self.model = QSVM(self.kernel)
        else:
            raise InvalidModelNameError(f'"{self.model_config.model_name}" is not implemented.')

    def build(self) -> BaseModel:
        """Build quantum model."""
        self._set_device()
        self._set_feature_map()
        self._set_kernel()
        self._set_model()

        return self.model
