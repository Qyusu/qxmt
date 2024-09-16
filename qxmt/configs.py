from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from qxmt.constants import MODULE_HOME


class PathConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: Path | str
    label: Path | str

    def model_post_init(self, __context: dict[str, Any]) -> None:
        if not Path(self.data).is_absolute():
            self.data = MODULE_HOME / self.data

        if not Path(self.label).is_absolute():
            self.label = MODULE_HOME / self.label


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["file", "generate"]
    path: Optional[PathConfig]
    params: Optional[dict[str, Any]] = None
    random_seed: int
    test_size: float = Field(ge=0.0, le=1.0)
    features: Optional[list[str]] = None
    raw_preprocess_logic: Optional[dict[str, Any]] = None
    transform_logic: Optional[dict[str, Any]] = None

    @model_validator(mode="before")
    def check_path_based_on_type(cls, values: dict[str, str]) -> dict[str, str]:
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


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path | str = ""
    description: str = ""
    dataset: DatasetConfig
    device: DeviceConfig
    feature_map: Optional[FeatureMapConfig] = None
    kernel: Optional[KernelConfig] = None
    model: ModelConfig
    evaluation: EvaluationConfig
