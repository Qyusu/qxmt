import types

from qxmt.configs import ExperimentConfig
from qxmt.devices.base import BaseDevice
from qxmt.devices.builder import DeviceBuilder
from qxmt.exceptions import InvalidModelNameError
from qxmt.feature_maps.base import BaseFeatureMap
from qxmt.kernels.base import BaseKernel
from qxmt.models.base import BaseMLModel
from qxmt.models.qsvm import QSVM
from qxmt.utils import load_object_from_yaml


class ModelBuilder:
    """Builder class for quantum machine learning models.
    This class is responsible for building quantum machine learning models.
    Absorb differences among various platforms, Feature Maps, and Kernels,
    and build models that can be handled as a common interface within the library.

    Examples:
        >>> from qxmt.configs import ExperimentConfig
        >>> from qxmt.models.builder import ModelBuilder
        >>> config = ExperimentConfig(path="configs/my_run.yaml")
        >>> model = ModelBuilder(config).build()
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize the model builder.

        Args:
            config (ExperimentConfig): Configuration for the experiment.
        """
        self.config: ExperimentConfig = config
        self.device: BaseDevice
        self.feature_map: BaseFeatureMap
        self.kernel: BaseKernel
        self.model: BaseMLModel

    def _set_device(self) -> None:
        """Set quantum device.

        Raises:
            InvalidPlatformError: platform is not implemented.
        """
        device_config = self.config.device
        self.device = DeviceBuilder(config=device_config).build()

    def _set_feature_map(self) -> None:
        """Set quantum feature map."""
        feature_map_config = self.config.feature_map
        if feature_map_config is None:
            return
        else:
            self.feature_map = load_object_from_yaml(
                config=feature_map_config.model_dump(),
                dynamic_params={
                    "n_qubits": self.config.device.n_qubits,
                },
            )

        if not (isinstance(self.feature_map, BaseFeatureMap) or isinstance(self.feature_map, types.FunctionType)):
            raise TypeError("Feature map must be a BaseFeatureMap instance or a function.")

    def _set_kernel(self) -> None:
        """Set quantum kernel."""
        kernel_config = self.config.kernel
        if kernel_config is None:
            return
        else:
            self.kernel = load_object_from_yaml(
                config=kernel_config.model_dump(),
                dynamic_params={
                    "device": self.device,
                    "feature_map": self.feature_map,
                },
            )

        if not isinstance(self.kernel, BaseKernel):
            raise TypeError("Kernel must be a BaseKernel instance.")

    def _set_model(self) -> None:
        """Set quantum model."""
        if self.config.model.name == "qsvm":
            self.model = QSVM(kernel=self.kernel, **self.config.model.params)
        else:
            raise InvalidModelNameError(f'"{self.config.model.name}" is not implemented.')

    def build(self) -> BaseMLModel:
        """
        Build quantum model by following steps:
            1. Set quantum device
            2. Set quantum feature map
            3. Set quantum kernel
            4. Set quantum model
        """
        self._set_device()
        self._set_feature_map()
        self._set_kernel()
        self._set_model()

        return self.model
