from pathlib import Path

from pydantic import BaseModel, ConfigDict

from qxmt.datasets.schema import Dataset
from qxmt.models.base import BaseMLModel


class RunRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: int
    desc: str
    execution_time: str
    commit_id: str
    config_path: Path | str
    evaluation: dict[str, float]


class RunArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    run_id: int
    dataset: Dataset
    model: BaseMLModel


class ExperimentDB(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    desc: str
    working_dirc: Path
    experiment_dirc: Path
    runs: list[RunRecord]
