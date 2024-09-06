from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from qxmt.constants import MODULE_HOME


class PathConfig(BaseModel):
    data: Path | str
    label: Path | str

    def model_post_init(self, __context: dict[str, Any]) -> None:
        if not Path(self.data).is_absolute():
            self.data = MODULE_HOME / self.data

        if not Path(self.label).is_absolute():
            self.label = MODULE_HOME / self.label


class DatasetConfig(BaseModel):
    type: Literal["file", "generate"]
    path: PathConfig
    random_seed: int
    test_size: float = Field(ge=0.0, le=1.0)
    features: Optional[list[str]] = None
    raw_preprocess_logic: Optional[dict[str, Any]] = None
    transform_logic: Optional[dict[str, Any]] = None


class DeviceConfig(BaseModel):
    platform: str
    name: str
    n_qubits: int
    shots: Optional[int] = None


class FeatureMapConfig(BaseModel):
    module_name: str
    implement_name: str
    params: Optional[dict[str, Any]] = None


class KernelConfig(BaseModel):
    module_name: str
    implement_name: str
    params: Optional[dict[str, Any]] = None


class ModelConfig(BaseModel):
    name: str
    file_name: str
    params: dict[str, Any]
    feature_map: Optional[FeatureMapConfig] = None
    kernel: Optional[KernelConfig] = None


class EvaluationConfig(BaseModel):
    default_metrics: list[str]


class ExperimentConfig(BaseModel):
    path: Path | str = ""
    description: str = ""
    dataset: DatasetConfig
    device: DeviceConfig
    feature_map: Optional[FeatureMapConfig] = None
    kernel: Optional[KernelConfig] = None
    model: ModelConfig
    evaluation: EvaluationConfig
