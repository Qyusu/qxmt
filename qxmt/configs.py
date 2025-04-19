import sys
from pathlib import Path
from typing import Any, Literal, Optional

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qxmt.constants import PROJECT_ROOT_DIR


class GlobalSettingsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    random_seed: int
    model_type: Literal["qkernel", "vqe"]
    task_type: Literal["classification", "regression"] | None = None

    @model_validator(mode="after")
    def check_task_type_for_kernel(self) -> Any:
        if self.model_type == "qkernel" and self.task_type is None:
            raise ValueError("task_type must be specified when model_type is 'kernel'")
        return self


class OpenMLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    id: Optional[int] = None
    return_format: str = "numpy"
    save_path: Optional[Path | str] = None

    def model_post_init(self, __context: dict[str, Any]) -> None:
        if (self.save_path is not None) and (not Path(self.save_path).is_absolute()):
            self.save_path = PROJECT_ROOT_DIR / self.save_path


class FileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_path: Path | str
    label_path: Optional[Path | str]
    label_name: Optional[str]

    def model_post_init(self, __context: dict[str, Any]) -> None:
        if not Path(self.data_path).is_absolute():
            self.data_path = PROJECT_ROOT_DIR / self.data_path

        if (self.label_path is not None) and not Path(self.label_path).is_absolute():
            self.label_path = PROJECT_ROOT_DIR / self.label_path


class GenerateDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generate_method: Literal["linear"]
    params: Optional[dict[str, Any]] = {}


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_ratio: float = Field(ge=0.0, le=1.0)
    validation_ratio: float = Field(ge=0.0, le=1.0, default=0.0)
    test_ratio: float = Field(ge=0.0, le=1.0)
    shuffle: bool = Field(default=True)

    @model_validator(mode="after")
    def check_ratio(self) -> Self:
        ratios = [self.train_ratio, self.validation_ratio, self.test_ratio]
        if sum(ratios) != 1:
            raise ValueError("The sum of the ratios must be 1.")
        return self


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    openml: Optional[OpenMLConfig] = None  # only need when use openml dataset
    file: Optional[FileConfig] = None  # only need when use file dataset
    generate: Optional[GenerateDataConfig] = None  # only need when use generated dataset
    split: SplitConfig
    features: Optional[list[str]] = None
    raw_preprocess_logic: Optional[list[dict[str, Any]] | dict[str, Any]] = None
    transform_logic: Optional[list[dict[str, Any]] | dict[str, Any]] = None


class HamiltonianConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module_name: str
    implement_name: str
    params: Optional[dict[str, Any]] = None


class DeviceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str
    device_name: str
    backend_name: Optional[str] = None
    n_qubits: int
    shots: Optional[int] = None
    random_seed: Optional[int] = None
    save_shots_results: bool = False

    @field_validator("shots")
    def check_shots(cls, value: int) -> int:
        if (value is not None) and (value < 1):
            raise ValueError("shots must be greater than or equal to 1")
        return value

    @model_validator(mode="after")
    def check_save_shots(self) -> Self:
        if (self.shots is None) and (self.save_shots_results):
            raise ValueError('The "shots" must be set to save the shot results.')
        return self


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


class AnsatzConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    module_name: str
    implement_name: str
    params: Optional[dict[str, Any]] = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    params: dict[str, Any]
    # Only need when model_type is "kernel"
    feature_map: Optional[FeatureMapConfig] = None
    kernel: Optional[KernelConfig] = None
    # Only need when model_type is "vqe"
    diff_method: Optional[str] = None
    optimizer_settings: Optional[dict[str, Any]] = None


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    default_metrics: list[str]
    custom_metrics: Optional[list[dict[str, Any]]] = None


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path | str = ""
    description: str = ""
    global_settings: GlobalSettingsConfig
    dataset: Optional[DatasetConfig] = None
    hamiltonian: Optional[HamiltonianConfig] = None
    device: DeviceConfig
    feature_map: Optional[FeatureMapConfig] = None
    kernel: Optional[KernelConfig] = None
    ansatz: Optional[AnsatzConfig] = None
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
            "hamiltonian": config.get("hamiltonian"),
            "device": config.get("device"),
            "feature_map": config.get("feature_map"),
            "kernel": config.get("kernel"),
            "ansatz": config.get("ansatz"),
            "model": config.get("model"),
            "evaluation": config.get("evaluation"),
        }
