from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from qxmt.constants import DEFAULT_METRICS_NAME, MODULE_HOME


class GlobalSettingsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    random_seed: int


class PathConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: Path | str
    label: Path | str

    def model_post_init(self, __context: dict[str, Any]) -> None:
        if not Path(self.data).is_absolute():
            self.data = MODULE_HOME / self.data

        if not Path(self.label).is_absolute():
            self.label = MODULE_HOME / self.label


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_ratio: float = Field(ge=0.0, le=1.0)
    validation_ratio: float = Field(ge=0.0, le=1.0, default=0.0)
    test_ratio: float = Field(ge=0.0, le=1.0)
    shuffle: bool = Field(default=True)

    @model_validator(mode="after")
    def check_ratio(self) -> "SplitConfig":
        ratios = [self.train_ratio, self.validation_ratio, self.test_ratio]
        if sum(ratios) != 1:
            raise ValueError("The sum of the ratios must be 1.")

        return self


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["file", "generate"]
    path: Optional[PathConfig]
    params: Optional[dict[str, Any]] = None
    random_seed: int
    split: SplitConfig
    features: Optional[list[str]] = None
    raw_preprocess_logic: Optional[dict[str, Any]] = None
    transform_logic: Optional[dict[str, Any]] = None

    @model_validator(mode="before")
    def check_path_based_on_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        type_ = values.get("type")
        path = values.get("path")

        if type_ == "file" and path is None:
            raise ValueError('path must be provided when type is "file".')

        return values


class DeviceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str
    name: str
    n_qubits: int
    shots: Optional[int] = None


class FeatureMapConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    module_name: str
    implement_name: str
    params: Optional[dict[str, Any]] = None


class KernelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    module_name: str
    implement_name: str
    params: Optional[dict[str, Any]] = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    file_name: str
    params: dict[str, Any]
    feature_map: Optional[FeatureMapConfig] = None
    kernel: Optional[KernelConfig] = None


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    default_metrics: list[str]
    custom_metrics: Optional[list[dict[str, Any]]] = None

    @model_validator(mode="after")
    def check_default_metrics(self) -> "EvaluationConfig":
        for metric_name in self.default_metrics:
            if metric_name not in DEFAULT_METRICS_NAME:
                raise ValueError(f"{metric_name} is not implemented.")
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path | str = ""
    description: str = ""
    global_settings: GlobalSettingsConfig
    dataset: DatasetConfig
    device: DeviceConfig
    feature_map: Optional[FeatureMapConfig] = None
    kernel: Optional[KernelConfig] = None
    model: ModelConfig
    evaluation: EvaluationConfig

    def __init__(self, **data: Any) -> None:
        """Initialize the experiment configuration.

        Case 1:
            Load the configuration from a file path.
            This case the data is a dictionary with a single key "path".
        Case 2:
            Load the configuration from a dictionary.
            This case the data is a dictionary with the configuration data.
        """
        if list(data.keys()) == ["path"]:
            config = self.load_from_path(data.get("path", ""))
            data.update(config)
        super().__init__(**data)

    def load_from_path(self, path: str) -> dict[str, Any]:
        with open(path, "r") as file:
            config = yaml.safe_load(file)

        if config is None:
            raise ValueError(f'The configuration file is empty. (path="{path}")')

        return {
            "description": config.get("description"),
            "global_settings": config.get("global_settings"),
            "dataset": config.get("dataset"),
            "device": config.get("device"),
            "feature_map": config.get("feature_map"),
            "kernel": config.get("kernel"),
            "model": config.get("model"),
            "evaluation": config.get("evaluation"),
        }
